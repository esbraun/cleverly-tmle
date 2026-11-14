"""The fitted result of adaptive TMLE for a randomized trial pooled with external data.

:func:`fit_trial_effect` runs :func:`~cleverly.estimators.atmle.fit_adaptive_tmle` for an
identified :class:`~cleverly.study.TrialATE` and wraps the engine's fit in
:class:`AdaptiveTMLEResult`. The engine module stays a numpy core: it reads arrays and
learners, and this module reads the study, the declared method and the data container.

The result declares the assessment family ``"augmented_trial"``. Its diagnostics read the
engine's own record: the overlap of the trial with the external rows, the score equations at
the end of the fit, the selected working models and the passes of the trial-membership
update. :mod:`cleverly._trial_family` holds the refusal every point-treatment sensitivity and
validation entry point raises for this family.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np

from .._inference_status import STATUS_RECORDS, WORKING_MODEL_STATUS, InferenceStatus
from .._summary import Detail, Summary, check_detail
from .._support import Support
from .._trial_family import AUGMENTED_TRIAL_FAMILY
from .._typing import FloatArray
from .._warnings import FitWarning
from ..exceptions import DataError, MethodConfigurationError
from ..inference.influence import ParameterEstimate
from ..inference.results import (
    estimate_curves,
    reported_status,
    select_estimates,
    smooth_contrast,
    sole_estimate,
)
from ..learners.crossfit import Folds, _fresh_seed
from ..learners.super_learner import resolve_learner
from ..provenance import Provenance, data_fingerprint, record
from ..utils.bounds import resolve_g_bounds
from .atmle import (
    SOURCE_FLOOR,
    AdaptiveTMLEFit,
    WorkingModel,
    fit_adaptive_tmle,
    stratified_partition,
)

if TYPE_CHECKING:
    from ..data.causal_data import CausalData
    from ..inference.delta import Transform
    from ..methods import AdaptiveTMLEMethod
    from ..study import IdentifiedEffect

__all__ = [
    "AdaptiveTMLECorrections",
    "AdaptiveTMLENuisanceReport",
    "AdaptiveTMLEResult",
    "AdaptiveTMLEScoreReport",
    "AdaptiveTMLESupport",
    "WorkingModelFold",
    "fit_trial_effect",
]


# ----------------------------------------------------------------------- the reports


@dataclass(frozen=True)
class AdaptiveTMLESupport:
    """How well the trial covers the external rows, and the treatment range in the trial.

    The effect averaged over every row is defined only where ``P(S = 1 | W) > 0`` (assumption
    A3). The source gives no floor, so the report reads the cross-fitted ``pi(W)`` on the
    external rows against :data:`~cleverly.estimators.atmle.SOURCE_FLOOR` and does not refuse.
    The effect averaged over the trial rows does not need A3, so ``adequate`` is then true
    and the shares are information only.

    Parameters
    ----------
    n : int
        Rows.
    n_trial : int
        Trial rows.
    n_external : int
        External rows.
    n_external_treated : int
        External rows with the treated arm.
    controls_only : bool
        Whether the external rows hold controls only.
    external_source_min : float
        The smallest cross-fitted ``P(S = 1 | W)`` on an external row, before bounds.
    external_source_quantiles : tuple of float
        Its 5%, 25% and 50% quantiles on the external rows.
    external_source_below_floor : float
        The share of external rows with ``P(S = 1 | W)`` below ``floor``.
    trial_row_treatment_range : tuple of float
        The smallest and largest bounded combined-data ``P(A = 1 | W)`` read on the trial
        rows.
    floor : float
        The reading floor of ``P(S = 1 | W)``.
    population : {"trial", "pooled"}
        The rows the effect is averaged over.
    """

    n: int
    n_trial: int
    n_external: int
    n_external_treated: int
    controls_only: bool
    external_source_min: float
    external_source_quantiles: tuple[float, float, float]
    external_source_below_floor: float
    trial_row_treatment_range: tuple[float, float]
    floor: float = SOURCE_FLOOR
    population: Literal["trial", "pooled"] = "pooled"

    @property
    def adequate(self) -> bool:
        """Whether A3 holds on the reading, which only the combined-rows effect needs."""
        return self.population == "trial" or self.external_source_below_floor == 0.0


@dataclass(frozen=True)
class AdaptiveTMLEScoreReport:
    """The score equations the fit solves, as left at its end.

    Each value is the absolute mean of one component of the influence function, pooled
    over the folds' selections. ``influence`` is gated by the fit's own stopping rule,
    ``sd(D) / (sqrt(n) log n)``; the three components are reported beside it.

    Parameters
    ----------
    effect_model : float
        ``|P_n m_A' U_A|``, the effect-model score.
    bias_model : float
        ``|P_n m_S' U_S|``, the bias-model score.
    source : float
        ``|P_n f (S - Pi*) C|``, the score of the trial-membership update.
    influence : float
        ``|P_n D|``, the mean of the whole influence function.
    tolerance : float
        The stopping rule's threshold for ``influence``.
    converged : bool
        Whether the fit met its stopping rule.
    """

    effect_model: float
    bias_model: float
    source: float
    influence: float
    tolerance: float
    converged: bool

    @property
    def passed(self) -> bool:
        """Whether ``influence`` is at or below ``tolerance``."""
        return self.influence <= self.tolerance


@dataclass(frozen=True)
class WorkingModelFold:
    """The working models one fold selected, and the size of each.

    Parameters
    ----------
    fold : int or None
        The outer fold whose training rows selected the models, or ``None`` for
        ``form="pooled"``.
    selection_rows : int
        The rows the models were selected on.
    effect_columns : tuple of str
        The selected effect-model columns.
    bias_columns : tuple of str
        The selected bias-model columns.
    effect_penalty : float
        The cross-validated lasso penalty of the effect model.
    bias_penalty : float
        The cross-validated lasso penalty of the bias model.
    effect_risk : float
        The smallest cross-validated risk on the effect model's penalty grid.
    bias_risk : float
        The smallest cross-validated risk on the bias model's penalty grid.
    """

    fold: int | None
    selection_rows: int
    effect_columns: tuple[str, ...]
    bias_columns: tuple[str, ...]
    effect_penalty: float
    bias_penalty: float
    effect_risk: float
    bias_risk: float


@dataclass(frozen=True)
class AdaptiveTMLENuisanceReport:
    """The learners the fit used, and the working models each fold selected.

    Parameters
    ----------
    learners : tuple of tuple of str
        ``(role, learner)`` pairs: outcome, treatment, trial membership and, with missing
        outcomes, missingness.
    folds : tuple of WorkingModelFold
        One record per selection: one per outer fold for ``form="cv"``, one for
        ``form="pooled"``.
    n_folds : int
        The outer folds of the fit.
    """

    learners: tuple[tuple[str, str], ...]
    folds: tuple[WorkingModelFold, ...]
    n_folds: int


@dataclass(frozen=True)
class AdaptiveTMLECorrections:
    """The updates the fit made after selecting the working models.

    Parameters
    ----------
    epsilon : tuple of float
        The trial-membership fluctuation of each pass.
    epsilon_effect : float or None
        The effect model's step along its target direction (``form="cv"``), or ``None``.
    epsilon_bias : tuple of float
        The bias model's step of each pass (``form="cv"``); empty for ``form="pooled"``.
    passes : int
        The passes of the trial-membership update.
    converged : bool
        Whether the passes met the stopping rule.
    """

    epsilon: tuple[float, ...]
    epsilon_effect: float | None
    epsilon_bias: tuple[float, ...]
    passes: int
    converged: bool


# ----------------------------------------------------------------------- the result


@dataclass
class AdaptiveTMLEResult:
    """The adaptive TMLE estimate of a trial's effect with external data.

    Behaves as a mapping from parameter name to
    :class:`~cleverly.inference.ParameterEstimate`, with the one name ``ate``. The estimate
    is the working-model projection of the trial effect, with status
    ``"working_model_projection"``: the interval is shown and covers the projection, and
    covers the trial effect itself only under the oracle-approximation condition of
    arXiv:2405.07186v2, Theorem 1.

    Parameters
    ----------
    estimates : dict of str to ParameterEstimate
        The estimate, keyed ``ate``.
    fit : AdaptiveTMLEFit
        The engine's fit: the influence curve, the working models and the targeting record.
    data : CausalData
        The prepared pooled data.
    identified_effect : IdentifiedEffect
        The identified question this fit answers.
    method : AdaptiveTMLEMethod
        The declared method.
    parameter_keys : dict of str to ParameterKey
        The structured identity of each estimate.
    provenance : Provenance
        The fit's record of versions, data and split.
    learners : dict of str to Any
        The resolved learner of each role.
    g_bounds : tuple of float
        The bounds the treatment probabilities were clipped to.
    fit_warnings : tuple of FitWarning
        The warnings the fit raised.

    Attributes
    ----------
    estimate : ParameterEstimate
    inference_status : str
    n : int
    influence_curves : dict
    working_models : tuple
    folds : Folds
    diagnostics : DiagnosticsFacade
    replayability : Replayability

    See Also
    --------
    cleverly.TrialWithExternalData : The design.
    cleverly.TrialATE : The estimand.
    cleverly.AdaptiveTMLEMethod : The method.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly import CausalStudy, TrialATE, TrialWithExternalData
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> s = rng.binomial(1, 0.5, n)
    >>> w = rng.normal(size=n)
    >>> a = np.where(s == 1, rng.binomial(1, 0.5, n), rng.binomial(1, 0.3, n))
    >>> y = 1.0 + a + w + 0.5 * (1 - s) + rng.normal(size=n)
    >>> frame = pd.DataFrame({"Y": y, "A": a, "W": w, "S": s})
    >>> study = CausalStudy(
    ...     frame,
    ...     design=TrialWithExternalData(outcome="Y", treatment="A", adjustment=("W",), source="S"),
    ... )
    >>> result = study.identify(TrialATE(population="trial")).estimate(
    ...     outcome_learner=LinearRegression(),
    ...     treatment_learner=LogisticRegression(),
    ...     n_folds=3,
    ...     random_state=0,
    ... )
    >>> result.inference_status
    'working_model_projection'
    >>> bool(abs(result.psi() - 1.0) < 0.5)
    True
    """

    estimates: dict[str, ParameterEstimate]
    fit: AdaptiveTMLEFit
    data: CausalData
    identified_effect: IdentifiedEffect
    method: AdaptiveTMLEMethod
    parameter_keys: dict[str, Any]
    provenance: Provenance
    learners: dict[str, Any] = field(default_factory=dict)
    g_bounds: tuple[float, float] = (0.0, 1.0)
    #: The warnings the fit raised (:mod:`cleverly._warnings`).
    fit_warnings: tuple[FitWarning, ...] = ()
    #: Assessment results keyed by operation and normalized arguments. ``init=False``, so a
    #: mapping handed to the constructor by ``dataclasses.replace`` cannot answer for
    #: another fit.
    assessment_cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    #: The assessment family: one declaration per assessment operation in
    #: :data:`cleverly.assessment.ASSESSMENT_CAPABILITIES`. The fit holds no point-treatment
    #: nuisance or targeting state, so the point declarations do not describe it.
    assessment_family: ClassVar[str] = AUGMENTED_TRIAL_FAMILY
    #: The method identity of the result.
    fitted_method: ClassVar[str] = "atmle"

    # ------------------------------------------------------------- mapping API

    def __getitem__(self, name: str) -> ParameterEstimate:
        try:
            return self.estimates[name]
        except KeyError:
            raise KeyError(
                f"{name!r} was not estimated; this fit reports {list(self.estimates)}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self.estimates)

    def __len__(self) -> int:
        return len(self.estimates)

    def __contains__(self, name: object) -> bool:
        return name in self.estimates

    def keys(self) -> Any:
        """The parameter names.

        Returns
        -------
        KeysView
            The names, in report order.
        """
        return self.estimates.keys()

    def values(self) -> Any:
        """The estimates.

        Returns
        -------
        ValuesView
            The estimates, in report order.
        """
        return self.estimates.values()

    def items(self) -> Any:
        """The ``(name, estimate)`` pairs.

        Returns
        -------
        ItemsView
            The pairs, in report order.
        """
        return self.estimates.items()

    # --------------------------------------------------------------- inference

    @property
    def estimate(self) -> ParameterEstimate:
        """The sole estimate."""
        return sole_estimate(self.estimates)

    @property
    def inference_status(self) -> InferenceStatus:
        """The inference status of the fit, ``"working_model_projection"``."""
        return reported_status(self.estimates)

    @property
    def n(self) -> int:
        """The number of rows, trial and external."""
        return self.data.n

    def psi(self, name: str | None = None) -> float:
        """Return a point estimate by parameter name.

        Parameters
        ----------
        name : str or None
            The parameter name. ``None`` takes the sole estimate.

        Returns
        -------
        float
            The point estimate.
        """
        return self.estimate.psi if name is None else self[name].psi

    @property
    def influence_curves(self) -> dict[str, FloatArray]:
        """The influence curves by parameter name."""
        return estimate_curves(self.estimates)

    def covariance(self, names: Sequence[str] | None = None) -> FloatArray:
        """Return the joint covariance of selected estimates.

        Parameters
        ----------
        names : sequence of str or None
            Names in output order. ``None`` selects all estimates.

        Returns
        -------
        ndarray
            The covariance of the estimators. The result holds one estimate, so this is its
            variance under the fit's own rule, the mean of the validation folds' variances
            for ``form="cv"``, which equals ``std_error ** 2``.
        """
        chosen = select_estimates(self.estimates, names)
        return np.array([[self.estimates[name].variance for name in chosen]], dtype=float)

    def contrast(
        self,
        function: Callable[[FloatArray], float],
        names: Sequence[str],
        *,
        name: str | None = None,
        gradient: Callable[[FloatArray], FloatArray] | None = None,
        transform: Transform | None = None,
    ) -> ParameterEstimate:
        """Return a smooth function of estimates with delta-method inference.

        Parameters
        ----------
        function : callable
            A smooth scalar function of the selected estimates.
        names : sequence of str
            The names passed to ``function`` in order.
        name : str or None
            The name of the derived estimate.
        gradient : callable or None
            The analytic gradient. ``None`` uses central differences.
        transform : Transform or None
            The map the interval is computed on.

        Returns
        -------
        ParameterEstimate
            The derived estimate, on the difference scale, with the status of its inputs.
        """
        return smooth_contrast(
            self.estimates,
            function,
            names,
            n=self.n,
            cluster=None,
            alpha=self.method.inference.alpha,
            name=name,
            scale="difference",
            gradient=gradient,
            transform=transform,
        )

    def ratio(self, numerator: str, denominator: str, **kwargs: Any) -> ParameterEstimate:
        """Refuse a ratio: only the difference scale is derived.

        Parameters
        ----------
        numerator : str
            The numerator's name.
        denominator : str
            The denominator's name.
        **kwargs : Any
            Ignored.

        Returns
        -------
        ParameterEstimate
            Never returns.

        Raises
        ------
        CapabilityError
            Always.
        """
        from ..exceptions import CapabilityError
        from ..study import TRIAL_RATIO_REFUSAL

        raise CapabilityError.from_refusal(TRIAL_RATIO_REFUSAL)

    # ---------------------------------------------------------- fit artifacts

    @property
    def working_models(self) -> tuple[WorkingModel, ...]:
        """The working models each fold selected, with their coefficients."""
        return self.fit.working_models

    @property
    def folds(self) -> Folds:
        """The outer split of the fit."""
        return self.fit.folds

    def support_report(self) -> AdaptiveTMLESupport:
        """The overlap of the trial with the external rows (A3) and the trial's arms.

        Returns
        -------
        AdaptiveTMLESupport
            The engine's record of both.
        """
        d = self.fit.diagnostics
        return AdaptiveTMLESupport(
            n=d.n,
            n_trial=d.n_trial,
            n_external=d.n_external,
            n_external_treated=d.n_external_treated,
            controls_only=d.controls_only,
            external_source_min=d.external_source_min,
            external_source_quantiles=d.external_source_quantiles,
            external_source_below_floor=d.external_source_below_floor,
            trial_row_treatment_range=d.trial_row_treatment_range,
            population=self.fit.population,
        )

    def score_report(self) -> AdaptiveTMLEScoreReport:
        """The score equations at the end of the fit, with the stopping rule's threshold.

        Returns
        -------
        AdaptiveTMLEScoreReport
            The four score means and the threshold.
        """
        scores = self.fit.scores
        return AdaptiveTMLEScoreReport(
            effect_model=scores.effect_model,
            bias_model=scores.bias_model,
            source=scores.source,
            influence=scores.influence,
            tolerance=scores.threshold,
            converged=self.fit.converged,
        )

    def nuisance_report(self) -> AdaptiveTMLENuisanceReport:
        """The learners and the working models each fold selected.

        Returns
        -------
        AdaptiveTMLENuisanceReport
            One record per selection.
        """
        from .._summary._learners import learner_name

        folds = tuple(
            WorkingModelFold(
                fold=model.fold,
                selection_rows=model.selection_rows,
                effect_columns=model.effect_columns,
                bias_columns=model.bias_columns,
                effect_penalty=float(model.effect.penalty),
                bias_penalty=float(model.bias.penalty),
                effect_risk=float(np.min(model.effect.risk)),
                bias_risk=float(np.min(model.bias.risk)),
            )
            for model in self.fit.working_models
        )
        return AdaptiveTMLENuisanceReport(
            learners=tuple(
                (role, learner_name(learner)) for role, learner in self.learners.items()
            ),
            folds=folds,
            n_folds=self.fit.folds.n_folds,
        )

    def corrections_report(self) -> AdaptiveTMLECorrections:
        """The updates after selection: the trial-membership passes and the model steps.

        Returns
        -------
        AdaptiveTMLECorrections
            The engine's targeting record.
        """
        fit = self.fit
        return AdaptiveTMLECorrections(
            epsilon=tuple(float(e) for e in fit.epsilon),
            epsilon_effect=None if fit.epsilon_effect is None else float(fit.epsilon_effect),
            epsilon_bias=tuple(float(e) for e in fit.epsilon_bias),
            passes=fit.passes,
            converged=fit.converged,
        )

    # ------------------------------------------------------------- assessment

    @property
    def diagnostics(self) -> Any:
        """The support, score, working-model and update diagnostics of the fit."""
        from ..assessment import DiagnosticsFacade

        return DiagnosticsFacade(self)

    def validate(self) -> Any:
        """Run the inexpensive default battery without refitting.

        Returns
        -------
        ValidationReport
            One item per validated check, read off the stored fit.
        """
        from ..assessment import validate_result

        return validate_result(self)

    def assess(
        self,
        *,
        include_refits: bool = False,
        include_retargets: bool = False,
        arguments: Mapping[str, Mapping[str, Any]] | None = None,
        random_state: int | None = None,
    ) -> Any:
        """Run the applicable post-fit assessment battery.

        Every sensitivity row refuses on this fit: no bound is derived for an adaptive
        working-model fit.

        Parameters
        ----------
        include_refits : bool
            Run requested operations that refit nuisance models.
        include_retargets : bool
            Include moderate retargets.
        arguments : mapping or None
            Per-operation keyword arguments.
        random_state : int or None
            Common seed for stochastic refit operations.

        Returns
        -------
        AssessmentReport
            Validation, diagnostics, and sensitivity in one report.
        """
        from ..assessment import assess_result

        return assess_result(
            self,
            include_refits=include_refits,
            include_retargets=include_retargets,
            arguments=arguments,
            random_state=random_state,
        )

    @property
    def sensitivity(self) -> Any:
        """The sensitivity-analysis facade, whose every row refuses on this fit."""
        from ..assessment import SensitivityFacade

        return SensitivityFacade(self)

    @property
    def replayability(self) -> Any:
        """Which assessment operations survive from the stored fit."""
        from ..assessment import replayability

        return replayability(self)

    # ------------------------------------------------------------- output

    def save(self, path: Any) -> Any:
        """Persist the fitted result to a trusted joblib artifact.

        The artifact records :data:`cleverly.__version__`. A load by a different version
        warns with :class:`~cleverly.exceptions.VersionMismatchWarning`.

        Parameters
        ----------
        path : path-like
            Destination file.

        Returns
        -------
        Path
            The destination path.
        """
        from .serialize import save as _save

        return _save(self, path)

    def to_frame(self) -> Any:
        """One row per reported parameter, in the backend the data came from.

        Returns
        -------
        dataframe
            The columns of a point-treatment report, plus ``population`` and ``form``.
        """
        rows = [estimate.to_dict() for estimate in self.estimates.values()]
        keys = list(dict.fromkeys(key for row in rows for key in row))
        payload: dict[str, list[Any]] = {
            key: [row.get(key, np.nan) for row in rows] for key in keys
        }
        payload["population"] = [self.fit.population for _ in rows]
        payload["form"] = [self.fit.form for _ in rows]
        return self.data.frame_like(payload)

    def summary(self, *, detail: Detail = "short") -> Summary:
        """The estimate, the trial and external rows, and what the interval covers.

        Parameters
        ----------
        detail : {"short", "full"}, default="short"
            ``"full"`` adds the assumptions and the settings.

        Returns
        -------
        Summary
            The summary. It prints as text and renders as HTML in a notebook.
        """
        from ._atmle_summary import atmle_summary

        return atmle_summary(self, check_detail(detail))

    def __repr__(self) -> str:
        from ._atmle_summary import atmle_repr

        return atmle_repr(self)

    def _repr_html_(self) -> str:
        return self.summary().as_html()


# ----------------------------------------------------------------------- the fit


def _support() -> Support:
    """The support of every adaptive TMLE estimate: experimental, with the shown reason.

    No support row reads an adaptive TMLE fit, so the label does not depend on the
    configuration. A row would not change it: the ``working_model_projection`` status is a
    shown status, and :func:`cleverly._support.support_for` keeps an estimate with a shown
    status experimental inside any row.
    """
    shown = STATUS_RECORDS[WORKING_MODEL_STATUS].shown_reason
    assert shown is not None
    return Support(
        "experimental",
        "No support row covers an adaptive TMLE fit. The registered study adaptive-tmle-trial "
        f"measured its intervals at its own simulated laws and settings only. {shown}",
        evidence=("F40",),
    )


def _outer_folds(data: Any, kwargs: dict[str, Any], seed: int, observed: Any) -> Folds:
    """Return the outer split: a supplied plan, one fold holding every row, or a stratified draw."""
    from ..methods import ATMLE_CV_FOLDS_REFUSAL

    plan = kwargs["split_plan"]
    if plan is not None:
        plan.verify(n=data.n, cluster=None)
        realized = plan.validate(
            n=data.n,
            treatment=data.treatment,
            source_fingerprint=data_fingerprint(data),
        )
        if len(realized) != 1:
            raise DataError(
                f"adaptive TMLE takes one outer split; the split plan holds {len(realized)}"
            )
        folds = realized[0]
        if kwargs["form"] == "cv" and folds.n_folds < 3:
            raise MethodConfigurationError.from_refusal(ATMLE_CV_FOLDS_REFUSAL)
        return folds
    if not kwargs["cross_fit"]:
        # Pooled form only (the method refuses cv without folds): every nuisance trains on
        # every row, as the engine does at one fold.
        return Folds.single(data.n)
    return stratified_partition(
        data.source.astype(float),
        data.outcome,
        observed,
        "binomial" if data.family == "binomial" else "gaussian",
        kwargs["n_folds"],
        seed,
    )


def fit_trial_effect(effect: IdentifiedEffect, method: AdaptiveTMLEMethod) -> AdaptiveTMLEResult:
    """Fit adaptive TMLE for an identified :class:`~cleverly.study.TrialATE`.

    ``None`` learners take the package's default library for the outcome family;
    ``g_bounds="auto"`` resolves at the number of rows.

    Parameters
    ----------
    effect : IdentifiedEffect
        The identified trial effect, bound to its study.
    method : AdaptiveTMLEMethod
        The declared method.

    Returns
    -------
    AdaptiveTMLEResult
        The fitted result.
    """
    from ..data.causal_data import CausalData
    from ..study import ParameterKey, TrialATE

    study = effect._study
    assert study is not None
    data = study.data
    estimand = effect.estimand
    assert isinstance(data, CausalData) and data.source is not None
    assert isinstance(estimand, TrialATE)
    kwargs = method.estimator_kwargs()
    seed = kwargs["random_state"]
    seed = _fresh_seed() if seed is None else int(seed)
    family: Literal["gaussian", "binomial"] = (
        "binomial" if data.family == "binomial" else "gaussian"
    )
    outcome_task = "classification" if family == "binomial" else "regression"
    inner = kwargs["inner_folds"]

    def resolved(spec: Any, task: Any) -> Any:
        return resolve_learner(spec, task=task, n_folds=inner, random_state=seed)

    learners: dict[str, Any] = {
        "outcome": resolved(kwargs["outcome_learner"], outcome_task),
        "treatment": resolved(kwargs["treatment_learner"], "classification"),
        "trial membership": resolved(kwargs["source_learner"], "classification"),
    }
    missing = effect.functional.missingness is not None
    if missing:
        learners["missingness"] = resolved(kwargs["missingness_learner"], "classification")
    g_bounds = resolve_g_bounds(kwargs["g_bounds"], data.n)
    split = _outer_folds(data, kwargs, seed, data.observed if missing else None)
    fit = fit_adaptive_tmle(
        data.outcome,
        data.treatment,
        data.source.astype(float),
        data.covariates,
        outcome_learner=learners["outcome"],
        treatment_learner=learners["treatment"],
        source_learner=learners["trial membership"],
        g_bounds=g_bounds,
        observed=data.observed if missing else None,
        missingness_learner=learners.get("missingness"),
        basis=kwargs["basis"],
        form=kwargs["form"],
        population=estimand.population,
        force_arm=kwargs["force_arm"],
        folds=split,
        inner_folds=inner,
        random_state=seed,
        outcome_family=family,
        covariate_names=data.covariate_names,
    )
    alpha = method.inference.alpha
    n = data.n
    estimate = ParameterEstimate(
        name="ate",
        psi=fit.estimate,
        influence_curve=fit.influence_curve,
        variance=float(fit.std_error**2),
        n=n,
        n_clusters=n,
        scale="difference",
        alpha=alpha,
        covariance_rule="centered" if fit.form == "cv" else "second_moment",
        inference=WORKING_MODEL_STATUS,
        support=_support(),
        outcome_family=family,
    )
    levels = tuple(data.treatment_levels)
    provenance = record(data, fit.folds, random_state=seed, run_id=method.runtime.run_id)
    from dataclasses import replace

    provenance = replace(
        provenance,
        protocol_fingerprint=None if effect.protocol is None else effect.protocol.fingerprint,
    )
    return AdaptiveTMLEResult(
        estimates={"ate": estimate},
        fit=fit,
        data=data,
        identified_effect=effect,
        method=method,
        parameter_keys={
            "ate": ParameterKey("ate", "ate", axis="arm", value=levels[1], reference=levels[0])
        },
        provenance=provenance,
        learners=learners,
        g_bounds=g_bounds,
    )
