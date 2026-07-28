r"""Did the targeting step actually work?

TMLE's guarantees follow from one thing: the estimator solves the efficient
influence-function equation,

.. math:: P_n D^*(\hat P)(O) = \frac{1}{n}\sum_i D^*_i = 0.

If that equation is not solved, the estimate is not the TMLE -- it is a plug-in
estimate with an unquantified bias, and its influence-curve standard error does not
describe it.  This is a *checkable* condition, unlike the identification assumptions,
and there is no reason not to check it on every fit.

The check compares :math:`|P_n D^*|` against the standard error.  The natural scale is
:math:`\widehat{se} / \sqrt{n}`: the score has to be small relative to the estimate's
own sampling variability, not merely small in absolute terms, since the units of the
outcome are arbitrary.  A converged fit typically lands near machine precision, many
orders of magnitude below the tolerance.

The usual reasons for a failure:

* **positivity.** Targeted predictions pin against ``[0, 1]`` and cannot move further,
  so the fluctuation cannot reach the root.  Check
  :meth:`~cleverly.sensitivity.SensitivityAnalysis.positivity`.
* **a fluctuation that has to travel far**, from a poor initial fit -- try
  ``targeting="one_step"``, which rebuilds the direction along the path.
* **too coarse a step** in the one-step walk -- lower ``step_size``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..estimators.base import format_table

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..estimators.base import TMLEResult

__all__ = ["ScoreCheck", "ScoreCheckRow", "score_check"]

#: The score must be below ``tolerance * se / sqrt(n)`` to pass.  A converged
#: fluctuation is normally near machine precision, so this is a loose gate that only
#: catches genuine failures.
DEFAULT_TOLERANCE = 1e-3


@dataclass(frozen=True)
class ScoreCheckRow:
    """The score diagnostic for one estimand or one targeted family."""

    name: str
    kind: str
    score: float
    threshold: float
    std_error: float
    passed: bool
    converged: bool
    n_iter: int
    method: str

    @property
    def ratio(self) -> float:
        """Score relative to its threshold; below 1 passes."""
        if self.threshold <= 0:
            return float("nan")
        return abs(self.score) / self.threshold


@dataclass(frozen=True)
class ScoreCheck:
    """Whether the targeting step solved the efficient score equation."""

    rows: tuple[ScoreCheckRow, ...]
    tolerance: float
    n: int

    @property
    def passed(self) -> bool:
        """True when every row is within tolerance."""
        return all(row.passed for row in self.rows)

    def __bool__(self) -> bool:
        return self.passed

    @property
    def failures(self) -> tuple[ScoreCheckRow, ...]:
        return tuple(row for row in self.rows if not row.passed)

    def to_frame(self, data: Any = None) -> Any:
        from ..utils.frames import frame_from_dict

        payload = {
            "name": [row.name for row in self.rows],
            "kind": [row.kind for row in self.rows],
            "score": [row.score for row in self.rows],
            "threshold": [row.threshold for row in self.rows],
            "ratio": [row.ratio for row in self.rows],
            "passed": [row.passed for row in self.rows],
            "n_iter": [row.n_iter for row in self.rows],
            "method": [row.method for row in self.rows],
        }
        if data is not None:
            return data.frame_like(payload)
        return frame_from_dict(payload)

    def summary(self) -> str:
        verdict = (
            "PASS: the targeting step solved the efficient score equation."
            if self.passed
            else "FAIL: the score equation was not solved -- the influence-curve standard "
            "errors below do not describe this estimate. See the module docstring for the "
            "usual causes."
        )
        return "\n".join(
            [
                "Score-equation check",
                "-" * 20,
                format_table(
                    ["target", "kind", "|score|", "threshold", "ratio", "ok"],
                    [
                        [
                            row.name,
                            row.kind,
                            f"{abs(row.score):.3e}",
                            f"{row.threshold:.3e}",
                            f"{row.ratio:.2e}",
                            "yes" if row.passed else "NO",
                        ]
                        for row in self.rows
                    ],
                ),
                "",
                verdict,
            ]
        )

    def raise_if_failed(self) -> None:
        """Raise when the check failed -- for use in pipelines that must not ship a bad fit."""
        if self.passed:
            return
        names = ", ".join(row.name for row in self.failures)
        raise AssertionError(
            f"the TMLE score equation was not solved for: {names}. {self.summary()}"
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


def score_check(result: TMLEResult, *, tolerance: float = DEFAULT_TOLERANCE) -> ScoreCheck:
    """Check that targeting solved the score equation.

    Two things are verified, and both matter:

    * per targeted family, the fluctuation's own score
      ``mean(w * h * (Y - Q*))`` -- this is the equation the fluctuation is
      *solving*;
    * per estimand, the mean of its influence curve -- this is the equation the
      *estimate* relies on, which for a derived estimand such as the risk ratio is a
      transformation of the first.
    """
    n = result.data.n
    rows: list[ScoreCheckRow] = []

    reference_se = max(
        (
            estimate.std_error
            for estimate in result.estimates.values()
            if np.isfinite(estimate.std_error)
        ),
        default=1.0,
    )

    for group, fluctuation in result.fluctuations.items():
        threshold = tolerance * reference_se / np.sqrt(n)
        score = fluctuation.score_norm
        rows.append(
            ScoreCheckRow(
                name=group,
                kind="fluctuation",
                score=score,
                threshold=float(threshold),
                std_error=float(reference_se),
                passed=bool(score <= threshold),
                converged=fluctuation.converged,
                n_iter=fluctuation.n_iter,
                method=fluctuation.method,
            )
        )

    for name, estimate in result.estimates.items():
        threshold = tolerance * estimate.std_error / np.sqrt(n)
        score = abs(estimate.score)
        rows.append(
            ScoreCheckRow(
                name=name,
                kind="influence curve",
                score=float(score),
                threshold=float(threshold),
                std_error=estimate.std_error,
                passed=bool(score <= threshold),
                converged=True,
                n_iter=0,
                method=estimate.scale,
            )
        )

    return ScoreCheck(rows=tuple(rows), tolerance=tolerance, n=n)
