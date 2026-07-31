"""What produced a number.

Two fits of "the same" analysis disagreeing is a routine and expensive question,
and answering it needs more than the settings: it needs to know whether the data
were the same bytes, whether the folds fell the same way, and which versions of
numpy and scikit-learn were in the environment.  :class:`Provenance` records
exactly that, cheaply, on every result.

What it deliberately does not record:

* **A git commit.**  A library must not assume it is being run from inside a
  repository, and shelling out to ``git`` on import is both slow and surprising.
  Pass ``run_id`` if you want to tie a result to something in your own system.
* **Anything that could reconstruct the data.**  The fingerprint is a 16-hex-digit
  BLAKE2b digest of the outcome, treatment and covariate bytes.  It answers "are
  these the same data?" and nothing else, so a result stays safe to attach to a
  ticket.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from .data.causal_data import CausalData
    from .learners.crossfit import Folds

__all__ = ["Provenance", "build", "fingerprint_array", "record"]

#: Length of the hex digests.  Eight bytes is far past what is needed to tell two
#: datasets apart in a workflow, and short enough to read out loud.
_DIGEST_BYTES = 8


def fingerprint_array(*arrays: Any) -> str:
    """A stable digest of one or more arrays' contents.

    Stable across processes and platforms of the same endianness, which is what
    makes it usable as an equality check between runs.  Shape is folded in so that
    the same values in a different layout do not collide.
    """
    hasher = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    for array in arrays:
        if array is None:
            hasher.update(b"<none>")
            continue
        values = np.ascontiguousarray(array)
        hasher.update(str(values.shape).encode())
        hasher.update(str(values.dtype).encode())
        hasher.update(values.tobytes())
    return hasher.hexdigest()


def _version(name: str) -> str:
    try:
        module = __import__(name)
    except ImportError:  # pragma: no cover - all are hard dependencies
        return "not installed"
    return str(getattr(module, "__version__", "unknown"))


@dataclass(frozen=True)
class Provenance:
    """Enough to tell whether two results came from the same place.

    Attributes
    ----------
    data_fingerprint:
        Digest of the outcome, treatment and covariates.  Equal fingerprints mean
        equal inputs; different ones mean the data moved, which is usually the
        answer when two runs disagree.
    fold_fingerprint:
        Digest of the realised fold assignment.  Recorded separately from
        ``random_state`` because folds are *not* recoverable from a seed alone --
        they also depend on row order, on the stratification variable, and on the
        scikit-learn version that generated them.

        Under ``repeats=R`` this covers *every* draw, in fit order.  A repeated fit is
        reproducible only if all ``R`` splits are, so a digest of one of them would be
        stating a guarantee the fit does not make.
    """

    cleverly_version: str
    python_version: str
    platform: str
    created_utc: str
    n: int
    n_covariates: int
    n_clusters: int | None
    data_fingerprint: str
    fold_fingerprint: str
    random_state: int | None = None
    run_id: str | None = None
    package_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Provenance:
        return cls(**payload)

    def describe(self) -> list[str]:
        """Lines for :meth:`~cleverly.TMLEResult.summary`."""
        lines = [
            f"data {self.data_fingerprint} | folds {self.fold_fingerprint} | "
            f"cleverly {self.cleverly_version} | {self.created_utc}"
        ]
        if self.run_id:
            lines.append(f"run_id: {self.run_id}")
        return lines


def build(
    *,
    n: int,
    n_covariates: int,
    n_clusters: int | None,
    data_fingerprint: str,
    fold_fingerprint: str,
    random_state: int | None = None,
    run_id: str | None = None,
) -> Provenance:
    """Stamp the environment onto a record whose data fields the caller has computed.

    :func:`record` reads a :class:`~cleverly.data.causal_data.CausalData`, and a
    longitudinal fit has no such thing -- its fingerprint has to cover every node rather
    than three arrays.  That is the *only* part that differs, so it is the only part
    passed in: the versions, the platform and the timestamp are the same question
    whatever the container, and are answered in one place so a second estimator cannot
    quietly ship a record with the package versions missing.
    """
    from ._version import __version__

    return Provenance(
        cleverly_version=__version__,
        python_version=sys.version.split()[0],
        platform=platform.platform(terse=True),
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n=n,
        n_covariates=n_covariates,
        n_clusters=n_clusters,
        data_fingerprint=data_fingerprint,
        fold_fingerprint=fold_fingerprint,
        random_state=random_state,
        run_id=run_id,
        package_versions={
            name: _version(name) for name in ("numpy", "scipy", "sklearn", "narwhals")
        },
    )


def record(
    data: CausalData,
    folds: Folds | Sequence[Folds],
    *,
    random_state: int | None = None,
    run_id: str | None = None,
) -> Provenance:
    """Build the provenance record for one fit.

    ``folds`` may be a single split or, under repeated cross-fitting, every draw in fit
    order -- all of which go into the one ``fold_fingerprint``.
    """
    from .learners.crossfit import Folds as _Folds

    draws = [folds] if isinstance(folds, _Folds) else list(folds)

    return build(
        n=data.n,
        n_covariates=len(data.covariate_names),
        n_clusters=None if data.cluster is None else int(np.unique(data.cluster).size),
        data_fingerprint=fingerprint_array(data.outcome, data.treatment, data.covariates),
        fold_fingerprint=fingerprint_array(*(draw.assignment for draw in draws)),
        random_state=random_state,
        run_id=run_id,
    )
