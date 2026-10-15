r"""A plain record of one fold's fitted treatment mechanism.

The omitted-variable bound under ``nu2_estimator="plugin"`` squares the fitted Riesz
representer, so its standard error needs the score of the model that produced
:math:`\hat g` (``docs/roadmap.md`` F26).  :func:`describe_mechanism` reads that model
inside the fold worker and returns a :class:`MechanismFit` of plain arrays and strings.  The
record survives a process pool and a pickle round trip, and it holds no fitted estimator.

The record describes; it does not judge.  Which records admit a stacked curve is the rule of
:func:`cleverly.sensitivity._mechanism_stack.plugin_mechanism_term`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .._typing import FloatArray

__all__ = ["MechanismFit", "ScalerStep", "describe_mechanism"]


@dataclass(frozen=True)
class ScalerStep:
    """One ``StandardScaler`` or ``"passthrough"`` step before the final model.

    Parameters
    ----------
    kind : {"scaler", "passthrough"}
        The step type.
    mean : ndarray or None
        The fitted ``mean_``, or ``None`` when the step does not center.
    scale : ndarray or None
        The fitted ``scale_``, or ``None`` when the step does not scale.
    """

    kind: Literal["scaler", "passthrough"]
    mean: FloatArray | None = None
    scale: FloatArray | None = None

    def transform(self, x: FloatArray) -> FloatArray:
        """Apply the fitted step to ``x``, as the pipeline did.

        Parameters
        ----------
        x : ndarray
            The ``(n, d)`` design before the step.

        Returns
        -------
        ndarray
            The design after the step.
        """
        out = np.asarray(x, dtype=float)
        if self.mean is not None:
            out = out - self.mean
        if self.scale is not None:
            out = out / self.scale
        return out


@dataclass(frozen=True)
class MechanismFit:
    """What one fold's fitted treatment learner was, as plain values.

    Parameters
    ----------
    kind : {"logistic", "constant", "other"}
        ``"logistic"`` when the final model is a scikit-learn ``LogisticRegression`` or
        ``LogisticRegressionCV``; ``"constant"`` when the fold's training rows held one
        treatment value, so the package fitted no model; ``"other"`` for any other learner.
    learner_name : str
        The class name of the final model.
    step_names : tuple of str
        The class names of the pipeline steps before the final model, in order
        (``"passthrough"`` for a passthrough step).
    steps : tuple of ScalerStep or None
        The fitted steps when every step before the final model is a ``StandardScaler`` or
        ``"passthrough"``, else ``None``.
    is_cv : bool
        Whether the final model is ``LogisticRegressionCV``.
    classes : ndarray or None
        The fitted ``classes_`` of a logistic model.
    coef : ndarray or None
        The fitted ``coef_``, ``(1, d)`` for two classes and ``(K, d)`` for ``K`` classes.
    intercept : ndarray or None
        The fitted ``intercept_``.
    fit_intercept : bool
        The model's ``fit_intercept``.
    C : float
        The inverse penalty strength; ``numpy.inf`` for no penalty.
    penalty : str or None
        The ``penalty`` parameter as set, with ``None`` kept as ``None``.
    l1_ratio : float or None
        The ``l1_ratio`` parameter as set.
    solver : str
        The ``solver`` parameter.
    class_weight : str or None
        ``repr`` of the ``class_weight`` parameter, or ``None`` when it is unset.
    n_iter : int
        The largest entry of the fitted ``n_iter_``.
    max_iter : int
        The model's ``max_iter``.
    """

    kind: Literal["logistic", "constant", "other"]
    learner_name: str
    step_names: tuple[str, ...] = ()
    steps: tuple[ScalerStep, ...] | None = ()
    is_cv: bool = False
    classes: FloatArray | None = None
    coef: FloatArray | None = None
    intercept: FloatArray | None = None
    fit_intercept: bool = True
    C: float = float("inf")
    penalty: str | None = None
    l1_ratio: float | None = None
    solver: str = ""
    class_weight: str | None = None
    n_iter: int = 0
    max_iter: int = 0


def describe_mechanism(model: Any) -> MechanismFit:
    """Record one fold's fitted treatment learner.

    A module-level function, so that a process pool can run it inside the fold worker,
    where the fitted model still exists.

    Parameters
    ----------
    model : object
        The fold's fitted learner, as :func:`~cleverly.estimators._nuisance.fit_on_rows`
        returned it, or the package's constant fit.

    Returns
    -------
    MechanismFit
        The plain record.
    """
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from ._nuisance import ConstantFit

    if isinstance(model, ConstantFit):
        return MechanismFit(kind="constant", learner_name="ConstantFit")
    final = model
    step_names: tuple[str, ...] = ()
    steps: tuple[ScalerStep, ...] | None = ()
    if isinstance(model, Pipeline):
        final = model.steps[-1][1]
        names: list[str] = []
        records: list[ScalerStep] = []
        for _, step in model.steps[:-1]:
            if step is None or step == "passthrough":
                names.append("passthrough")
                records.append(ScalerStep(kind="passthrough"))
            elif type(step) is StandardScaler:
                names.append("StandardScaler")
                mean = getattr(step, "mean_", None)
                scale = getattr(step, "scale_", None)
                records.append(
                    ScalerStep(
                        kind="scaler",
                        mean=None
                        if mean is None or not step.with_mean
                        else np.array(mean, dtype=float),
                        scale=None
                        if scale is None or not step.with_std
                        else np.array(scale, dtype=float),
                    )
                )
            else:
                names.append(type(step).__name__)
        step_names = tuple(names)
        steps = tuple(records) if len(records) == len(names) else None
    learner_name = type(final).__name__
    if type(final) not in (LogisticRegression, LogisticRegressionCV):
        return MechanismFit(
            kind="other", learner_name=learner_name, step_names=step_names, steps=steps
        )
    params = final.get_params(deep=False)
    class_weight = params.get("class_weight")
    n_iter = np.asarray(getattr(final, "n_iter_", [0]))
    C = (
        float(getattr(final, "C_", [params.get("C", 1.0)])[0])
        if type(final) is LogisticRegressionCV
        else float(params.get("C", 1.0))
    )
    penalty = params.get("penalty")
    return MechanismFit(
        kind="logistic",
        learner_name=learner_name,
        step_names=step_names,
        steps=steps,
        is_cv=type(final) is LogisticRegressionCV,
        classes=np.array(final.classes_, dtype=float),
        coef=np.array(final.coef_, dtype=float),
        intercept=np.array(final.intercept_, dtype=float).reshape(-1),
        fit_intercept=bool(params.get("fit_intercept", True)),
        C=C,
        penalty=None if penalty is None else str(penalty),
        l1_ratio=None if params.get("l1_ratio") is None else float(params["l1_ratio"]),
        solver=str(params.get("solver", "")),
        class_weight=None if class_weight is None else repr(class_weight),
        n_iter=int(n_iter.max()) if n_iter.size else 0,
        max_iter=int(params.get("max_iter", 0)),
    )
