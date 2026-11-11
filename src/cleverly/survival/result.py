"""The result of a continuous-time survival fit."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._support import Support
from ..inference.influence import ParameterEstimate, make_estimate
from ..utils.frames import emit_frame

#: Until the validation study registers a support row, every estimate is experimental.
UNVALIDATED: Support = Support(
    "experimental",
    "No validation study covers the continuous-time survival TMLE yet.",
)

if TYPE_CHECKING:
    from ._targeting import WalkResult
    from .engine import ContinuousSurvivalRecord

__all__ = ["ContinuousSurvivalResult", "build_estimates"]


def _t(time: float) -> str:
    return f"{time:g}"


def level_name(arm: str, cause: str | None, time: float) -> str:
    """The parameter name of one arm's risk.

    Parameters
    ----------
    arm : str
        The arm label.
    cause : str or None
        The cause label, ``None`` with one event type.
    time : float
        The target time.

    Returns
    -------
    str
        ``risk_regimen[arm @ t=time]`` or ``cif_regimen[arm, cause @ t=time]``.
    """
    if cause is None:
        return f"risk_regimen[{arm} @ t={_t(time)}]"
    return f"cif_regimen[{arm}, {cause} @ t={_t(time)}]"


def contrast_name(head: str, treated: str, reference: str, cause: str | None, time: float) -> str:
    """The parameter name of a contrast of the two arms.

    Parameters
    ----------
    head : str
        ``"ate_regimen"`` or ``"rr_regimen"``.
    treated : str
        The treated arm's label.
    reference : str
        The reference arm's label.
    cause : str or None
        The cause label, ``None`` with one event type.
    time : float
        The target time.

    Returns
    -------
    str
        The name.
    """
    inner = f"{treated} vs {reference}" + ("" if cause is None else f", {cause}")
    return f"{head}[{inner} @ t={_t(time)}]"


def build_estimates(
    walk: WalkResult,
    arms: tuple[str, str],
    causes: Mapping[int, str],
    horizons: tuple[float, ...],
    *,
    n: int,
    alpha: float,
) -> dict[str, ParameterEstimate]:
    """Assemble the risks, risk differences and risk ratios from the targeted walk.

    Parameters
    ----------
    walk : WalkResult
        The targeting outcome.
    arms : tuple of str
        ``(reference, treated)`` labels, arm codes 0 and 1.
    causes : mapping of int to str
        The causes in target order.
    horizons : tuple of float
        The target times.
    n : int
        The number of units.
    alpha : float
        Significance level.

    Returns
    -------
    dict of str to ParameterEstimate
        Every arm's risk, then each difference and ratio, by time.
    """
    labels = list(causes.values())
    J, K = len(labels), len(horizons)
    single = J == 1
    psi = walk.evaluation.psi
    ic = walk.evaluation.influence
    out: dict[str, ParameterEstimate] = {}
    for a in (0, 1):
        for j, cause in enumerate(labels):
            for k, time in enumerate(horizons):
                p = (a * J + j) * K + k
                name = level_name(arms[a], None if single else cause, time)
                out[name] = make_estimate(
                    name,
                    psi[p],
                    ic[:, p],
                    n=n,
                    scale="level",
                    alpha=alpha,
                    bounds=(0.0, 1.0),
                    support=UNVALIDATED,
                )
    for j, cause in enumerate(labels):
        for k, time in enumerate(horizons):
            p0, p1 = j * K + k, (J + j) * K + k
            label = None if single else cause
            name = contrast_name("ate_regimen", arms[1], arms[0], label, time)
            out[name] = make_estimate(
                name,
                psi[p1] - psi[p0],
                ic[:, p1] - ic[:, p0],
                n=n,
                scale="difference",
                alpha=alpha,
                bounds=(-1.0, 1.0),
                support=UNVALIDATED,
            )
            name = contrast_name("rr_regimen", arms[1], arms[0], label, time)
            if psi[p0] > 0 and psi[p1] > 0:
                log_ic = ic[:, p1] / psi[p1] - ic[:, p0] / psi[p0]
                out[name] = make_estimate(
                    name,
                    psi[p1] / psi[p0],
                    log_ic,
                    n=n,
                    scale="ratio",
                    alpha=alpha,
                    log_psi=float(np.log(psi[p1] / psi[p0])),
                    support=UNVALIDATED,
                )
    return out


@dataclass(frozen=True)
class ContinuousSurvivalResult(Mapping[str, ParameterEstimate]):
    """The targeted risks of a continuous-time survival fit.

    Behaves as a mapping from parameter name to
    :class:`~cleverly.inference.ParameterEstimate`.

    Parameters
    ----------
    estimates : dict of str to ParameterEstimate
        Each arm's risk (``risk_regimen[arm @ t=60]``, or ``cif_regimen[arm, cause @ t=60]``
        with competing causes), the risk differences (``ate_regimen[...]``) and the risk ratios
        (``rr_regimen[...]``).
    record : ContinuousSurvivalRecord
        The nuisance selections and the targeting record.
    simultaneous : SimultaneousBands or None
        The sup-t band over every risk and risk difference, when requested.
    arms : tuple of str
        ``(reference, treated)``.
    causes : tuple of str
        The cause labels.
    horizons : tuple of float
        The target times.
    n : int
        The number of units.
    backend : str or None
        The dataframe library the data arrived in; :meth:`to_frame` returns a frame of it.
    """

    estimates: dict[str, ParameterEstimate]
    record: ContinuousSurvivalRecord
    simultaneous: Any
    arms: tuple[str, str]
    causes: tuple[str, ...]
    horizons: tuple[float, ...]
    n: int
    backend: str | None = None

    def __getitem__(self, name: str) -> ParameterEstimate:
        return self.estimates[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.estimates)

    def __len__(self) -> int:
        return len(self.estimates)

    @property
    def n_folds(self) -> int:
        """The number of cross-fitting folds, ``1`` for a full-sample fit."""
        return int(self.record.n_folds)

    def summary(self, *, detail: Literal["short", "full"] = "short") -> Any:
        """A printable report of the fit.

        Parameters
        ----------
        detail : {"short", "full"}
            ``"full"`` adds the assumptions and how the update ended.

        Returns
        -------
        Summary
            The report.
        """
        from .._summary._continuous_survival import continuous_survival_summary

        return continuous_survival_summary(self, detail=detail)

    def _repr_html_(self) -> str:
        return str(self.summary().as_html())

    def __repr__(self) -> str:
        return (
            f"ContinuousSurvivalResult(parameters={len(self.estimates)}, n={self.n}, "
            f"causes={len(self.causes)}, times={len(self.horizons)})"
        )

    def to_frame(self) -> Any:
        """One row per parameter, in the backend the data arrived in.

        Returns
        -------
        dataframe
            The estimates with their standard errors and intervals.
        """
        rows = [estimate.to_dict() for estimate in self.estimates.values()]
        columns: list[str] = []
        for row in rows:
            columns.extend(key for key in row if key not in columns)
        return emit_frame(
            {key: [row.get(key) for row in rows] for key in columns}, backend=self.backend
        )

    def save(self, path: Any) -> Any:
        """Write this result to a trusted joblib artifact; see :func:`cleverly.load`.

        Parameters
        ----------
        path : path-like
            Destination file.

        Returns
        -------
        Path
            The destination path, as given.
        """
        from ..estimators.serialize import save as _save

        return _save(self, path)
