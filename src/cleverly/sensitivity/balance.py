"""Covariate balance: how alike treated and untreated units look before and after weighting.

The standardized mean difference (SMD) of a covariate is the difference between its mean in
one treatment group and in the other, divided by a common spread. Inverse-propensity
weighting aims to make the groups alike on every covariate, so an SMD that stays large after
weighting says the treatment model missed something about that covariate. This is a
description of the data and the fitted weights, not an estimate: it carries no interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._refusal import RefusalReason
from .._typing import FloatArray
from ..exceptions import CapabilityError, DataError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .._summary import Detail, Summary
    from ..estimators.base import TMLEResult

__all__ = ["BalanceRow", "CovariateBalance", "covariate_balance"]

#: The conventional size below which a difference counts as small.
DEFAULT_THRESHOLD = 0.1


@dataclass(frozen=True)
class BalanceRow:
    """One covariate in one contrast.

    Parameters
    ----------
    contrast : str
        The two groups compared, in words.
    covariate : str
        The covariate; one level of a categorical covariate has its own row.
    mean_group : float
        The covariate's mean in the first group, with the data's weights only.
    mean_rest : float
        Its mean in the other group, with the data's weights only.
    mean_group_weighted : float
        The first group's mean under the fit's inverse-propensity weights.
    mean_rest_weighted : float
        The other group's mean under those weights.
    spread : float
        The pooled standard deviation both differences are divided by, from the two groups
        before weighting by the propensity. ``nan`` for a covariate with no spread.
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
        The contrasts, in order: the treated group against the untreated group for a
        two-arm treatment, each arm against the other arms otherwise.
    n : int
        The units with a recorded treatment.
    n_repeats : int
        The cross-fitting repeats the fit holds; the first one's propensities are used.
    limits : tuple of float or None
        The limits the fit put on the propensities, which the weights respect.
    weighted_data : bool
        Whether the data carried observation weights, which both columns use.
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
    reference level gets a row too.
    """
    names = list(data.covariate_names)
    columns = [np.asarray(data.covariates[:, index], dtype=float) for index in range(len(names))]
    position = {name: index for index, name in enumerate(names)}
    for encoding in data.encodings:
        kept = [position[name] for name in encoding.generated if name in position]
        if len(kept) != len(encoding.generated):
            continue  # a screened-out indicator: the reference level cannot be restored
        names.append(f"{encoding.column}__{encoding.dropped_level}")
        columns.append(1.0 - np.sum([columns[index] for index in kept], axis=0))
    return names, np.column_stack(columns) if columns else np.empty((data.n, 0))


def covariate_balance(
    result: TMLEResult, *, threshold: float = DEFAULT_THRESHOLD
) -> CovariateBalance:
    """Compute each covariate's standardized mean difference, before and after weighting.

    Use it to check the treatment model on the covariates you know. A weighted difference
    near zero says the weights made the groups look alike on that covariate; one that is
    still large says the model missed something about it. The weights are the fit's own:
    the data's weight divided by the propensity of the arm each unit received, after the
    limits the fit put on the propensity. Each difference is divided by the pooled standard
    deviation of the two groups before the propensity weighting, so the two columns are
    comparable.

    With two arms the treated group is compared with the untreated group. With more arms
    each arm is compared with all the others, and each other unit keeps the weight of its
    own arm. A categorical covariate gets one row per level.

    The propensities are the first cross-fitting repeat's out-of-fold predictions. The
    result is descriptive: it has no interval or test, and a small difference on the
    covariates listed does not show balance on covariates that were not.

    Parameters
    ----------
    result : TMLEResult
        A fitted point-treatment result with an estimated or declared treatment mechanism.
    threshold : float, default=0.1
        The absolute difference the summary and the plot hold each covariate against.

    Returns
    -------
    CovariateBalance
        The differences, with ``to_frame()``, ``summary()`` and ``plot()``.

    Raises
    ------
    CapabilityError
        When the fit estimated no treatment mechanism.
    DataError
        When the treatment is continuous, which has no groups to compare.

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
    """
    from ..targets.population_intervention import NATURAL_COURSE_SUPPORT_REFUSAL

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
    propensity = result.repeats[0].nuisance.propensity
    bounds = getattr(result.config, "g_bounds", None)
    limits = None if bounds is None else (float(bounds[0]), float(bounds[1]))
    values = (
        np.asarray(propensity.values, dtype=float)
        if limits is None
        else np.asarray(propensity.truncate(limits).values, dtype=float)
    )
    treatment = np.asarray(data.treatment, dtype=float)
    recorded = np.isfinite(treatment)
    arms = tuple(float(arm) for arm in propensity.arms)
    # Each recorded unit's weight is the data's weight over the probability of its own arm.
    own = np.empty(int(recorded.sum()))
    for arm in arms:
        mask = treatment[recorded] == arm
        own[mask] = values[recorded, propensity.column_for(arm)][mask]
    base = np.asarray(data.weights, dtype=float)[recorded]
    inverse = base / own
    names, matrix = _covariate_columns(data)
    matrix = matrix[recorded]
    name = str(data.treatment_name)

    def words(code: float) -> str:
        return f"{name} = {data.arm_label(code)}"

    if len(arms) == 2:
        pairs = [(arms[1], f"{words(arms[1])} vs {words(arms[0])}")]
    else:
        pairs = [(arm, f"{words(arm)} vs the other arms") for arm in arms]
    held = treatment[recorded]
    rows: list[BalanceRow] = []
    for arm, contrast in pairs:
        inside = held == arm
        outside = ~inside if len(arms) > 2 else held == arms[0]
        for index, covariate in enumerate(names):
            column = matrix[:, index]
            raw = [_weighted_moments(column[mask], base[mask]) for mask in (inside, outside)]
            weighted = [
                _weighted_moments(column[mask], inverse[mask]) for mask in (inside, outside)
            ]
            variance = (raw[0][1] + raw[1][1]) / 2.0
            spread = (
                float(np.sqrt(variance)) if np.isfinite(variance) and variance > 0 else float("nan")
            )
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
        contrasts=tuple(contrast for _, contrast in pairs),
        n=int(recorded.sum()),
        n_repeats=len(result.repeats),
        limits=limits,
        weighted_data=bool(np.ptp(base) > 0.0),
        threshold=float(threshold),
    )
