"""Covariate balance: how alike treated and untreated units look before and after weighting.

The standardized mean difference (SMD) of a covariate is the difference between its mean in
one treatment group and in the other, divided by a standard deviation. Inverse-propensity
weighting aims to make the groups alike on every covariate, so an SMD that stays large after
weighting says the treatment model missed something about that covariate. This is a
description of the data and the fitted weights, not an estimate: it carries no interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._refusal import RefusalReason
from .._trial_family import refuse_augmented_trial
from .._typing import FloatArray
from ..exceptions import CapabilityError, DataError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .._summary import Detail, Summary
    from ..estimators.base import TMLEResult

__all__ = ["BalanceRow", "CovariateBalance", "covariate_balance"]

#: The conventional size below which a difference counts as small.
DEFAULT_THRESHOLD = 0.1

#: The estimands whose weights the balance can rebuild, keyed to the fluctuation group that
#: targets them: the arm means (the ATE and every estimand built from arm means) and the
#: two conditional effects.
BalanceEstimand = Literal["ate", "att", "atc"]
_GROUPS: dict[str, str] = {"ate": "mean", "att": "att", "atc": "atc"}


@dataclass(frozen=True)
class BalanceRow:
    """One covariate in one contrast.

    Parameters
    ----------
    contrast : str
        The two groups compared, in words.
    covariate : str
        The covariate; one level of a categorical covariate has its own row, named
        ``"column = level"``.
    mean_group : float
        The covariate's mean in the first group, with the data's weights only.
    mean_rest : float
        Its mean in the other group, with the data's weights only.
    mean_group_weighted : float
        The first group's mean under the fit's weights.
    mean_rest_weighted : float
        The other group's mean under those weights.
    spread : float
        The standard deviation both differences are divided by, computed before weighting
        by the propensity (see :attr:`CovariateBalance.estimand`). ``nan`` for a covariate
        with no spread.
    smd_unweighted : float
        The standardized mean difference before weighting.
    smd_weighted : float
        The standardized mean difference after weighting.
    """

    contrast: str
    covariate: str
    mean_group: float
    mean_rest: float
    mean_group_weighted: float
    mean_rest_weighted: float
    spread: float
    smd_unweighted: float
    smd_weighted: float


@dataclass(frozen=True)
class CovariateBalance:
    """The standardized mean differences of a fit's covariates, before and after weighting.

    Parameters
    ----------
    rows : tuple of BalanceRow
        One row per contrast and covariate.
    treatment : str
        The treatment's name.
    contrasts : tuple of str
        The contrasts, in order. For the ATE: the treated group against the untreated group
        for a two-arm treatment, each arm against the rest otherwise. For the ATT or ATC:
        each non-reference arm against the reference arm.
    n : int
        The units with a recorded treatment.
    n_repeats : int
        The cross-fitting repeats the fit holds; the first one's propensities are used.
    limits : tuple of float or None
        The limits the fit put on the propensities for this estimand, which the weights
        respect.
    weighted_data : bool
        Whether the data carried observation weights, which both columns use.
    estimand : {"ate", "att", "atc"}
        Whose weights and spread the differences use. ``"ate"``: each unit weighted by the
        inverse of the probability of its own arm, divided by the pooled standard deviation
        of every arm. ``"att"``: the arm the effect is among keeps its units' weights and
        the reference arm is weighted by the odds of that arm, divided by that arm's
        standard deviation. ``"atc"``: the mirror image, among the reference arm.
    cross_fitted : bool
        Whether the propensities are out-of-fold predictions; in-sample otherwise.
    n_arms : int
        The treatment's arms.
    threshold : float
        The size an absolute difference is held against.
    """

    rows: tuple[BalanceRow, ...]
    treatment: str
    contrasts: tuple[str, ...]
    n: int
    n_repeats: int
    limits: tuple[float, float] | None
    weighted_data: bool
    estimand: BalanceEstimand = "ate"
    cross_fitted: bool = False
    n_arms: int = 2
    threshold: float = DEFAULT_THRESHOLD

    def to_frame(self, backend: str | None = None) -> Any:
        """Return one row per contrast and covariate.

        Parameters
        ----------
        backend : {"pandas", "polars", "pyarrow"} or None, default=None
            Dataframe backend to return. ``None`` uses pandas when installed, then the first
            available backend.

        Returns
        -------
        dataframe
            The columns ``contrast``, ``covariate``, the four group means, ``spread``,
            ``smd_unweighted`` and ``smd_weighted``.
        """
        from ..utils.frames import frame_from_dict

        names = (
            "contrast",
            "covariate",
            "mean_group",
            "mean_rest",
            "mean_group_weighted",
            "mean_rest_weighted",
            "spread",
            "smd_unweighted",
            "smd_weighted",
        )
        return frame_from_dict(
            {name: [getattr(row, name) for row in self.rows] for name in names}, backend=backend
        )

    def rows_of(self, contrast: str) -> tuple[BalanceRow, ...]:
        """The rows of one contrast.

        Parameters
        ----------
        contrast : str
            One of :attr:`contrasts`.

        Returns
        -------
        tuple of BalanceRow
            Its covariates, in the covariates' order.
        """
        return tuple(row for row in self.rows if row.contrast == contrast)

    def summary(self, detail: Detail = "short") -> Summary:
        """Return the printed summary of the balance.

        Parameters
        ----------
        detail : {"short", "full"}, default="short"
            ``"short"`` prints each covariate's difference before and after weighting.
            ``"full"`` adds the four group means and the spread each difference divides by.

        Returns
        -------
        Summary
            A ``str`` whose value is the printed text. Its ``facts``, ``tables``, ``notes``
            and ``reading`` hold the parts as data.
        """
        from .._summary import check_detail
        from .._summary._balance import balance_summary

        return balance_summary(self, check_detail(detail))

    def plot(self, **kwargs: Any) -> Any:
        """Plot each covariate's absolute difference before and after weighting.

        Parameters
        ----------
        **kwargs
            Passed to :func:`cleverly.plotting.balance`: ``max_covariates=``,
            ``treatment_label=``, ``labels=`` and ``title=``.

        Returns
        -------
        Figure
            One panel per contrast.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.linear_model import LinearRegression, LogisticRegression
        >>> from cleverly import ATE, CausalStudy, PointTreatment
        >>> from cleverly.datasets import make_linear_ate
        >>> from cleverly.sensitivity import covariate_balance
        >>> frame, _ = make_linear_ate(n=200, seed=0)
        >>> study = CausalStudy(
        ...     frame,
        ...     design=PointTreatment(
        ...         outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")
        ...     ),
        ... )
        >>> result = study.identify(ATE()).estimate(
        ...     outcome_learner=LinearRegression(),
        ...     treatment_learner=LogisticRegression(max_iter=1000),
        ...     cross_fit=False,
        ...     random_state=0,
        ... )
        >>> fig = covariate_balance(result).plot()
        >>> fig.axes[1].get_xlabel()
        'Absolute standardized mean difference'
        >>> plt.close(fig)
        """
        from ..plotting import balance

        return balance(self, **kwargs)

    def _repr_html_(self) -> str:
        return self.summary().as_html()

    def __repr__(self) -> str:
        from .._summary import short_repr

        return short_repr(
            "CovariateBalance",
            (
                ("treatment", self.treatment),
                ("estimand", self.estimand),
                ("covariates", len({row.covariate for row in self.rows})),
                ("contrasts", len(self.contrasts)),
            ),
        )


def _weighted_moments(values: FloatArray, weights: FloatArray) -> tuple[float, float]:
    """The weighted mean and the reliability-weighted sample variance of one column.

    With unit weights these are the ordinary mean and the sample variance. The variance
    divides by ``sum(w) - sum(w^2) / sum(w)``, which is what makes it unbiased for weights
    that mean how much a row counts rather than how often it occurred.
    """
    total = float(weights.sum())
    mean = float(np.sum(weights * values) / total)
    spread = total - float(np.sum(weights**2)) / total
    if spread <= 0.0:
        return mean, float("nan")
    return mean, float(np.sum(weights * (values - mean) ** 2) / spread)


def _covariate_columns(data: Any) -> tuple[list[str], FloatArray]:
    """The covariates by column, with every level of a categorical covariate in its own.

    The data holds a categorical covariate as indicators with one level left out as the
    reference; the left-out level's indicator is one minus the others, restored here so the
    reference level gets a row too. A two-level covariate keeps its one indicator: the
    restored level would be its mirror image. Levels are named ``"column = level"``.
    """
    names = list(data.covariate_names)
    columns = [np.asarray(data.covariates[:, index], dtype=float) for index in range(len(names))]
    position = {name: index for index, name in enumerate(names)}
    for encoding in data.encodings:
        kept = [position[name] for name in encoding.generated if name in position]
        if len(encoding.generated) == 1 and encoding.generated[0] == encoding.column:
            continue  # a boolean column: one indicator under its own name
        levels = [level for level in encoding.levels if level != encoding.dropped_level]
        for name, level in zip(encoding.generated, levels, strict=True):
            if name in position:
                names[position[name]] = f"{encoding.column} = {level}"
        if len(kept) != len(encoding.generated) or len(kept) == 1:
            continue  # a screened-out indicator, or a mirror image of the one kept
        names.append(f"{encoding.column} = {encoding.dropped_level}")
        columns.append(1.0 - np.sum([columns[index] for index in kept], axis=0))
    return names, np.column_stack(columns) if columns else np.empty((data.n, 0))


def _reported_groups(result: TMLEResult) -> set[str]:
    """The fluctuation groups of the estimands the fit reports."""
    from ..targets import TARGETS, parameter_stem

    names = tuple(getattr(result.config, "estimands", ())) or tuple(result.estimates)
    found = set()
    for name in names:
        target = TARGETS.get(str(name)) or TARGETS.get(parameter_stem(str(name)))
        if target is not None:
            found.add(target.group)
    return found


def _choose_estimand(result: TMLEResult, estimand: str | None) -> BalanceEstimand:
    groups = _reported_groups(result)
    if estimand is not None:
        if estimand not in _GROUPS:
            raise ValueError(f"estimand must be 'ate', 'att' or 'atc'; got {estimand!r}")
        if _GROUPS[estimand] not in groups:
            raise DataError(
                f"the fit reports no {estimand} estimate, so it computed no {estimand} "
                "weights to check the balance of.",
                reason=RefusalReason.DIFFERENT_ESTIMAND,
                alternative=f"Refit with identify({estimand.upper()}()) to check its weights.",
            )
        return estimand  # type: ignore[return-value]
    for name, group in _GROUPS.items():
        if group in groups:
            return name  # type: ignore[return-value]
    raise CapabilityError(
        "the fit reports no estimand weighted by the probability of the arm each unit "
        f"received (its targets belong to {sorted(groups)}), so the balance under those "
        "weights would describe a weighting the fit did not use.",
        reason=RefusalReason.DIFFERENT_ESTIMAND,
        alternative="Check the support of this fit with diagnostics.support().",
    )


def covariate_balance(
    result: TMLEResult,
    *,
    estimand: BalanceEstimand | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> CovariateBalance:
    """Compute each covariate's standardized mean difference, before and after weighting.

    Use it to check the treatment model on the covariates you know. A weighted difference
    near zero says the weights made the groups look alike on that covariate; one that is
    still large says the model missed something about it. The weights are the fit's own for
    the chosen estimand, after the limits the fit put on the propensity for it:

    - ``"ate"``: each unit counts as the data's weight divided by the probability of the arm
      it received, within ``g_bounds``. Each difference is divided by the pooled standard
      deviation, the square root of the average of every arm's variance before weighting
      (Austin 2009; cobalt's ``s.d.denom = "pooled"``). With more than two arms this is one
      spread per covariate, the same in every panel. Each arm is compared with all the
      others together, so imbalances of opposite sign in two other arms can cancel; cobalt
      compares pairs instead.
    - ``"att"``: the effect among an arm's own units. That arm's units keep the data's
      weights and the reference arm's units count as their weight times the odds of that
      arm against the reference, within ``g_bounds_conditional``. Each difference is divided
      by the standard deviation in that arm, as cobalt does for the ATT.
    - ``"atc"``: the mirror image, among the reference arm, whose standard deviation is the
      spread.

    The spread is computed before the propensity weighting, so the two columns of a row
    divide by the same number. Indicators of a binary covariate and of each level of a
    categorical one are standardized the same way; cobalt's default prints their raw
    difference in proportions instead. A categorical covariate gets one row per level, a
    two-level one a single row.

    The propensities are the first cross-fitting repeat's out-of-fold predictions, or the
    in-sample predictions of a fit that was not cross-fitted. The result is descriptive: it
    has no interval or test, and a small difference on the covariates listed does not show
    balance on covariates that were not.

    Parameters
    ----------
    result : TMLEResult
        A fitted point-treatment result with an estimated or declared treatment mechanism.
    estimand : {"ate", "att", "atc"} or None, default=None
        Whose weights to use. It must be one the fit reports; the first of ``"ate"``,
        ``"att"`` and ``"atc"`` that it reports when omitted.
    threshold : float, default=0.1
        The absolute difference the summary and the plot hold each covariate against.

    Returns
    -------
    CovariateBalance
        The differences, with ``to_frame()``, ``summary()`` and ``plot()``.

    Raises
    ------
    CapabilityError
        When the fit estimated no treatment mechanism, or reports no estimand weighted by
        the probability of each unit's own arm.
    DataError
        When the treatment is continuous, which has no groups to compare, or the fit
        reports no estimate of ``estimand``.
    ValueError
        When ``threshold`` is not positive or ``estimand`` is not one of the three.

    See Also
    --------
    cleverly.sensitivity.positivity_report : Overlap of the same propensities, as numbers.
    cleverly.plotting.overlap : The propensities' distribution, as a figure.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly import ATE, CausalStudy, PointTreatment
    >>> from cleverly.datasets import make_linear_ate
    >>> from cleverly.sensitivity import covariate_balance
    >>> frame, _ = make_linear_ate(n=200, seed=0)
    >>> study = CausalStudy(
    ...     frame,
    ...     design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")),
    ... )
    >>> result = study.identify(ATE()).estimate(
    ...     outcome_learner=LinearRegression(),
    ...     treatment_learner=LogisticRegression(max_iter=1000),
    ...     cross_fit=False,
    ...     random_state=0,
    ... )
    >>> balance = covariate_balance(result)
    >>> balance.to_frame()["covariate"].tolist()
    ['W1', 'W2', 'W3', 'W4']
    >>> balance.estimand, balance.cross_fitted
    ('ate', False)
    """
    from ..targets.population_intervention import NATURAL_COURSE_SUPPORT_REFUSAL

    refuse_augmented_trial(result, "covariate_balance", "report")
    _refuse_sequential(result)
    if not result.nuisance.fits_treatment:
        raise CapabilityError.from_refusal(NATURAL_COURSE_SUPPORT_REFUSAL)
    data = result.data
    if data.is_continuous_treatment:
        raise DataError(
            f"{data.treatment_name} is continuous, so there are no treatment groups to "
            "compare and the balance has no rows to fill.",
            reason=RefusalReason.DIFFERENT_ESTIMAND,
            alternative="Check the density ratio of the shift with diagnostics.support().",
        )
    if not float(threshold) > 0.0:
        raise ValueError(f"threshold must be positive; got {threshold!r}")
    chosen = _choose_estimand(result, estimand)
    propensity = result.repeats[0].nuisance.propensity
    config = result.config
    bounds = getattr(config, "g_bounds" if chosen == "ate" else "g_bounds_conditional", None)
    limits = None if bounds is None else (float(bounds[0]), float(bounds[1]))
    values = (
        np.asarray(propensity.values, dtype=float)
        if limits is None
        else np.asarray(propensity.truncate(limits).values, dtype=float)
    )
    treatment = np.asarray(data.treatment, dtype=float)
    recorded = np.isfinite(treatment)
    held = treatment[recorded]
    probabilities = values[recorded]
    arms = tuple(float(arm) for arm in propensity.arms)
    reference = float(getattr(config, "reference_arm", arms[0]))
    if reference not in arms:
        reference = arms[0]

    def probability_of(arm: float) -> FloatArray:
        return probabilities[:, propensity.column_for(arm)]

    # The probability of each recorded unit's own arm.
    own = np.empty(held.shape[0])
    for arm in arms:
        mask = held == arm
        own[mask] = probability_of(arm)[mask]
    base = np.asarray(data.weights, dtype=float)[recorded]
    names, matrix = _covariate_columns(data)
    matrix = matrix[recorded]
    name = str(data.treatment_name)

    def words(code: float) -> str:
        return f"{name} = {data.arm_label(code)}"

    # One comparison: the units of the first group, of the second, their weights after the
    # propensity weighting, and the arms whose variances make the spread.
    comparisons: list[tuple[str, Any, Any, FloatArray, tuple[float, ...]]] = []
    others = [arm for arm in arms if arm != reference]
    if chosen == "ate":
        inverse = base / own
        if len(arms) == 2:
            (other,) = others
            comparisons.append(
                (
                    f"{words(other)} vs {words(reference)}",
                    held == other,
                    held == reference,
                    inverse,
                    arms,
                )
            )
        else:
            comparisons.extend(
                (f"{words(arm)} vs the rest", held == arm, held != arm, inverse, arms)
                for arm in arms
            )
    else:
        for arm in others:
            focal = arm if chosen == "att" else reference
            # The focal arm's units keep their weight; every other unit counts as the odds
            # of the focal arm against its own.
            tilted = base * probability_of(focal) / own
            tilted = np.where(held == focal, base, tilted)
            comparisons.append(
                (
                    f"{words(arm)} vs {words(reference)}",
                    held == arm,
                    held == reference,
                    tilted,
                    (focal,),
                )
            )
    variances = {
        arm: np.asarray(
            [
                _weighted_moments(matrix[held == arm, index], base[held == arm])[1]
                for index in range(len(names))
            ]
        )
        for arm in arms
    }
    rows: list[BalanceRow] = []
    for contrast, inside, outside, weighted_by, spread_arms in comparisons:
        variance = np.mean([variances[arm] for arm in spread_arms], axis=0)
        for index, covariate in enumerate(names):
            column = matrix[:, index]
            raw = [_weighted_moments(column[mask], base[mask]) for mask in (inside, outside)]
            weighted = [
                _weighted_moments(column[mask], weighted_by[mask]) for mask in (inside, outside)
            ]
            value = float(variance[index])
            spread = float(np.sqrt(value)) if np.isfinite(value) and value > 0 else float("nan")
            rows.append(
                BalanceRow(
                    contrast=contrast,
                    covariate=covariate,
                    mean_group=raw[0][0],
                    mean_rest=raw[1][0],
                    mean_group_weighted=weighted[0][0],
                    mean_rest_weighted=weighted[1][0],
                    spread=spread,
                    smd_unweighted=(raw[0][0] - raw[1][0]) / spread,
                    smd_weighted=(weighted[0][0] - weighted[1][0]) / spread,
                )
            )
    return CovariateBalance(
        rows=tuple(rows),
        treatment=name,
        contrasts=tuple(item[0] for item in comparisons),
        n=int(recorded.sum()),
        n_repeats=len(result.repeats),
        limits=limits,
        weighted_data=bool(np.ptp(base) > 0.0),
        estimand=chosen,
        cross_fitted=bool(getattr(config, "cross_fit", False)),
        n_arms=len(arms),
        threshold=float(threshold),
    )


def _refuse_sequential(result: Any) -> None:
    """Refuse the fits whose treatment is not one point-treatment model with its weights."""
    from .._refusal import NO_ALTERNATIVE, RefusalReason
    from ..exceptions import CapabilityError
    from ..longitudinal.estimator import LongitudinalResult
    from ..survival.result import ContinuousSurvivalResult

    if isinstance(result, LongitudinalResult):
        what = (
            "a longitudinal fit, whose weights multiply a treatment and a censoring model "
            "at each node"
        )
    elif isinstance(result, ContinuousSurvivalResult):
        what = (
            "a continuous-time survival fit, whose weights also carry a censoring model over time"
        )
    else:
        return
    raise CapabilityError(
        "covariate_balance weighs the covariates by a point-treatment fit's treatment model; "
        f"it does not weigh {what} yet.",
        reason=RefusalReason.IMPLEMENTATION_GAP,
        alternative=NO_ALTERNATIVE,
        roadmap="X100",
    )
