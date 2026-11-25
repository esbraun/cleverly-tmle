"""Checkpoints and progress for the long draws of :func:`~cleverly.validation.refute` (X106 G3).

The five opt-in tests with long default draw counts (``dummy_outcome``, ``simulated_outcome``,
``plasmode``, ``bootstrap_stability`` and ``bootstrap_measurement_error``) run every draw from
its own seed descriptor, never from the shared generator of the call. A draw's record is
therefore a function of the fit, the request, the library versions and its seed alone, so a
draw stored by an interrupted call is the draw an uninterrupted call computes, and a resume
that loads it and runs the rest returns the same report bit for bit.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import json
import os
import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..exceptions import CapabilityError, RefusalReason

if TYPE_CHECKING:
    from .refute import EmpiricalRefitRecord
    from .simulation import ReplicationFailure

__all__ = ["RefutationCheckpoint", "RefutationProgress"]

#: The tests a checkpoint stores. The default tests draw from the call's shared generator and
#: run 20 draws, so a checkpoint does not cover them.
CHECKPOINTED_TESTS = (
    "dummy_outcome",
    "simulated_outcome",
    "plasmode",
    "bootstrap_stability",
    "bootstrap_measurement_error",
)

_REQUEST = "request.json"
_CANDIDATES = "candidates.json"
_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")


@dataclass(frozen=True)
class RefutationProgress:
    """How far one checkpointed test of a :func:`~cleverly.validation.refute` call has run.

    Parameters
    ----------
    test : str
        The test, as ``tests=`` names it.
    completed : int
        Draws finished so far, loaded ones included.
    requested : int
        Draws the test runs.
    failed : int
        Finished draws whose refit failed so far. A ratio draw that refits to a
        nonpositive value is moved to the failures after the last draw, so it is not
        counted here.
    resumed : int
        Finished draws loaded from the checkpoint rather than refitted.
    candidate : str or None
        The candidate of a variable-importance run whose fit this is, or ``None``.
    """

    test: str
    completed: int
    requested: int
    failed: int
    resumed: int
    candidate: str | None = None


@dataclass(frozen=True)
class RefutationCheckpoint:
    """Keep each finished draw of the long refutation tests as a file in a directory.

    Pass it as ``refute(..., checkpoint=RefutationCheckpoint(directory))``. Each finished draw of
    ``dummy_outcome``, ``simulated_outcome``, ``plasmode``, ``bootstrap_stability`` and
    ``bootstrap_measurement_error`` is written as one JSON file, under a temporary name and then
    renamed, so a call stopped mid-write never reads a partial file. Run the same call again
    with the same directory and it loads the stored draws, refits only the missing ones, and
    returns the report an uninterrupted call returns. The placebo, noise and subset tests are
    not stored.

    The directory records the request on first use: the estimator's class and settings, the
    data's fingerprint, the resolved seed and repeat seeds, every estimate and standard error of
    the fit, the estimand, each stored test's draw count, process or declaration, bands, rule
    and resampling, and the versions of ``cleverly``, NumPy, SciPy and scikit-learn. A call
    whose request differs in any part refuses, naming the part. Two development commits can
    share one ``cleverly`` version, so give each commit its own directory. An unseeded fit
    draws its seed at the first call; the directory records it, and a resume passes it back as
    ``random_state=``. A directory that holds no draw yet takes a new request. A facade call
    whose answer is already cached returns it without reading the draws. Run one call per
    directory at a time: the request file is not locked.

    Parameters
    ----------
    directory : str or os.PathLike
        Where the files go. It is created when missing.
    """

    directory: str | os.PathLike[str]

    def child(self, name: str, *, record: bool = True) -> RefutationCheckpoint:
        """The checkpoint of one candidate of a variable-importance run.

        Parameters
        ----------
        name : str
            The candidate's name. Its directory is named by a digest of it, so a name holding
            ``/`` or ``..`` stays inside this directory.
        record : bool
            Whether to record the name in this directory. The preview of planned refits
            passes ``False`` and writes nothing.

        Returns
        -------
        RefutationCheckpoint
            A checkpoint in a subdirectory of this one.

        Raises
        ------
        CapabilityError
            If two names map to one subdirectory.
        """
        digest = hashlib.sha256(name.encode()).hexdigest()[:16]
        root = Path(self.directory)
        path = root / _CANDIDATES
        mapping: dict[str, str] = json.loads(path.read_text()) if path.exists() else {}
        owner = mapping.get(digest)
        if owner is not None and owner != name:
            raise CapabilityError(
                f"the checkpoint directory {str(root)!r} maps candidates {owner!r} and {name!r} "
                "to one subdirectory",
                reason=RefusalReason.INVALID_DECLARATION,
                alternative="Pass a fresh checkpoint directory.",
            )
        if owner is None and record:
            mapping[digest] = name
            _write_json(path, mapping)
        return RefutationCheckpoint(root / f"candidate-{digest}")

    def _path(self, test: str, replicate: int) -> Path:
        return Path(self.directory) / test / f"draw-{replicate:09d}.json"

    def load(self, test: str, replicate: int) -> EmpiricalRefitRecord | ReplicationFailure | None:
        """Return the stored draw ``replicate`` of ``test``, or ``None`` when there is none.

        Parameters
        ----------
        test : str
            The test.
        replicate : int
            The draw's index.

        Returns
        -------
        EmpiricalRefitRecord, ReplicationFailure or None
            The stored draw, or ``None``.
        """
        path = self._path(test, replicate)
        if not path.exists():
            return None
        return _from_json(json.loads(path.read_text()))

    def save(self, test: str, replicate: int, outcome: Any) -> None:
        """Store draw ``replicate`` of ``test``.

        Parameters
        ----------
        test : str
            The test.
        replicate : int
            The draw's index.
        outcome : EmpiricalRefitRecord or ReplicationFailure
            What the draw returned.
        """
        _write_json(self._path(test, replicate), _to_json(test, outcome))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True))
    os.replace(temporary, path)


def _to_json(test: str, outcome: Any) -> dict[str, Any]:
    from .refute import EmpiricalRefitRecord

    fields = {
        field.name: (
            int(value)
            if isinstance(value, np.integer)
            else float(value)
            if isinstance(value, np.floating)
            else value
        )
        for field in dataclasses.fields(outcome)
        for value in (getattr(outcome, field.name),)
    }
    kind = "record" if isinstance(outcome, EmpiricalRefitRecord) else "failure"
    return {"test": test, "kind": kind, "fields": fields}


def _from_json(payload: Mapping[str, Any]) -> Any:
    from .refute import EmpiricalRefitRecord
    from .simulation import ReplicationFailure

    cls = EmpiricalRefitRecord if payload["kind"] == "record" else ReplicationFailure
    return cls(**payload["fields"])


# ------------------------------------------------------------------ the request key


def _refuse(directory: Any, message: str, alternative: str) -> CapabilityError:
    return CapabilityError(
        f"the checkpoint directory {str(directory)!r} {message}",
        reason=RefusalReason.INVALID_DECLARATION,
        alternative=alternative,
    )


def config_key(value: Any, *, path: str = "value", seen: frozenset[int] = frozenset()) -> Any:
    """A process-stable key of a configuration object, refusing what has no stable key.

    Arrays hash by content and scalars stand for themselves, as in the assessment cache. A
    learner keys by class and ``get_params(deep=True)``; a dataclass by its fields; a partial by
    its function and arguments; a function by module and qualname; anything else by class only
    when its repr holds no memory address.

    Parameters
    ----------
    value : Any
        The object.
    path : str
        Where the object sits, for the refusal's message.
    seen : frozenset of int
        Ids of the objects already on the walk, so a cycle ends.

    Returns
    -------
    Any
        A JSON-serializable key.

    Raises
    ------
    CapabilityError
        If a part of ``value`` is a lambda or local function, or an object whose only
        description holds a memory address.
    """
    from .._assessment_cache import _normalize

    if isinstance(value, np.ndarray) and value.dtype == object:
        # An object array holds references, which ``_normalize`` cannot hash by content.
        return {
            "object array": list(value.shape),
            "items": config_key(value.ravel().tolist(), path=path, seen=seen),
        }
    if value is None or isinstance(value, (str, int, float, bool, np.ndarray, np.generic)):
        return _normalize(value)
    if id(value) in seen:
        return {"cycle": type(value).__qualname__}
    seen = seen | {id(value)}
    if isinstance(value, Mapping):
        return {
            str(key): config_key(item, path=f"{path}[{key!r}]", seen=seen)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [
            config_key(item, path=f"{path}[{index}]", seen=seen) for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        return sorted(
            json.dumps(config_key(item, path=path, seen=seen), sort_keys=True) for item in value
        )
    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    if isinstance(value, functools.partial):
        return {
            "partial": config_key(value.func, path=f"{path}.func", seen=seen),
            "args": config_key(value.args, path=f"{path}.args", seen=seen),
            "keywords": config_key(value.keywords, path=f"{path}.keywords", seen=seen),
        }
    if inspect.isclass(value):
        return {"class": f"{value.__module__}.{value.__qualname__}"}
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value):
        qualname = getattr(value, "__qualname__", "")
        if "<lambda>" in qualname or "<locals>" in qualname:
            raise _refuse(
                "",
                f"cannot key {path}, the function {qualname!r}: a lambda or a local function "
                "has no name that survives a new process",
                "Define the function at module level, or call refute() without a checkpoint.",
            )
        return {"function": f"{getattr(value, '__module__', '')}.{qualname}"}
    get_params = getattr(value, "get_params", None)
    if callable(get_params):
        try:
            params = get_params(deep=True)
        except TypeError:
            params = get_params()
        return {"type": kind, "params": config_key(params, path=f"{path}.params", seen=seen)}
    if dataclasses.is_dataclass(value):
        return {
            "type": kind,
            "fields": {
                field.name: config_key(
                    getattr(value, field.name), path=f"{path}.{field.name}", seen=seen
                )
                for field in dataclasses.fields(value)
            },
        }
    text = repr(value)
    if _ADDRESS.search(text):
        raise _refuse(
            "",
            f"cannot key {path}, a {type(value).__qualname__} whose only description holds a "
            "memory address, so its key would change in every process",
            "Give the object a repr without an address, or call refute() without a checkpoint.",
        )
    return {"type": kind, "repr": text}


def _estimator_key(estimator: Any) -> Any:
    if estimator is None:
        return None
    try:
        parameters: Mapping[str, Any] = inspect.signature(type(estimator).__init__).parameters
    except (TypeError, ValueError):
        parameters = {}
    settings = {
        name: config_key(getattr(estimator, name), path=f"estimator.{name}")
        for name in parameters
        if name != "self" and hasattr(estimator, name)
    }
    return {"class": f"{type(estimator).__module__}.{type(estimator).__qualname__}", **settings}


def _data_key(result: Any) -> Any:
    from .._assessment_cache import _normalize

    provenance = getattr(result, "provenance", None)
    fingerprint = getattr(provenance, "data_fingerprint", None)
    data = getattr(result, "data", None)
    if fingerprint is None and data is not None:
        try:
            from ..provenance import data_fingerprint

            fingerprint = data_fingerprint(data)
        except Exception:
            fingerprint = None
    extras = {
        name: _normalize(np.asarray(value))
        for name in ("weights", "cluster")
        for value in (getattr(data, name, None),)
        if value is not None
    }
    return {"fingerprint": fingerprint, **extras}


def _estimates_key(result: Any) -> Any:
    out = {}
    for name in result:
        estimate = result[name]
        out[str(name)] = [
            float(estimate.psi).hex(),
            float(getattr(estimate, "plugin_std_error", float("nan"))).hex(),
        ]
    return out


def _versions() -> dict[str, str]:
    import scipy
    import sklearn

    from .. import __version__

    return {
        "cleverly": __version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def request_parts(
    result: Any,
    *,
    estimand: str,
    seed: int | None,
    repeat_seeds: Any,
    sampling: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The parts of a call's request that every stored test shares.

    Parameters
    ----------
    result : Any
        The fit.
    estimand : str
        The estimand the call refits.
    seed : int or None
        The call's resolved root seed, or ``None`` for the preview of a call that draws one.
    repeat_seeds : Any
        The fit's repeat seeds, or the structural fit seeds.
    sampling : mapping or None
        The limit, rule, unit count and seed of a sampled call (X106 part H), or ``None`` for
        a call on the full data. A sampled call's draws are refits of the sampled baseline, so a
        directory of one never serves the other.

    Returns
    -------
    dict
        One key per part, each a JSON-serializable value.
    """
    return {
        "estimator": _estimator_key(getattr(result, "estimator", None)),
        "data": _data_key(result),
        "seeds": {
            "root": None if seed is None else int(seed),
            "repeats": config_key(repeat_seeds, path="repeat seeds"),
        },
        "estimates": _estimates_key(result),
        "estimand": estimand,
        "library versions": _versions(),
        "sampling": None if sampling is None else dict(sampling),
    }


_PART_NAMES = {
    "estimator": "estimator",
    "data": "data",
    "seeds": "seeds",
    "estimates": "estimates",
    "estimand": "request",
    "library versions": "library versions",
    "sampling": "sampling",
}


def stored_root(checkpoint: RefutationCheckpoint) -> int | None:
    """The root seed the directory's request records, or ``None`` when it holds none.

    Parameters
    ----------
    checkpoint : RefutationCheckpoint
        The directory.

    Returns
    -------
    int or None
        The recorded root seed.
    """
    path = Path(checkpoint.directory) / _REQUEST
    if not path.exists():
        return None
    root = json.loads(path.read_text()).get("common", {}).get("seeds", {}).get("root")
    return None if root is None else int(root)


def seed_refusal(checkpoint: RefutationCheckpoint, root: int, resolved: int | None) -> Exception:
    """The refusal of a call whose root seed differs from the one the directory records.

    Parameters
    ----------
    checkpoint : RefutationCheckpoint
        The directory.
    root : int
        The recorded root seed.
    resolved : int or None
        This call's root seed, or ``None`` when it draws one at the call.

    Returns
    -------
    CapabilityError
        The refusal.
    """
    drawn = "draws a fresh one" if resolved is None else f"resolves {resolved}"
    return _refuse(
        checkpoint.directory,
        f"was written by a call with root seed {root}, and this call {drawn}",
        f"Pass random_state={root} to resume it, or use a fresh directory.",
    )


class CheckpointSession:
    """One call's view of a checkpoint directory: its request checked, its draws loaded.

    Parameters
    ----------
    checkpoint : RefutationCheckpoint
        The directory.
    parts : dict
        :func:`request_parts` of the call.
    progress : callable or None
        The caller's progress callback.
    read_only : bool
        Whether the session only reads the directory, as the preview of planned refits does.
        It checks the request the same way and writes nothing.
    drawn : bool
        Whether the call draws its root seed with fresh entropy, which a refusal names.
    """

    def __init__(
        self,
        checkpoint: RefutationCheckpoint | None,
        parts: dict[str, Any],
        progress: Callable[[RefutationProgress], Any] | None,
        *,
        read_only: bool = False,
        drawn: bool = False,
    ) -> None:
        self.checkpoint = checkpoint
        self.progress = progress
        self.read_only = read_only
        self.drawn = drawn
        self.parts = parts
        self.reference: dict[str, Any] | None = None
        if checkpoint is not None:
            self._check_common()

    def bind_reference(self, reference: dict[str, Any]) -> None:
        """Key every stored test on a sampled call's baseline as well (X106 part H).

        A sampled call refits its baseline on each call, and every stored draw is compared
        with it. The baseline is not in the common request, which the preview opens before any
        fit, so it enters each test's parts here: a resume whose baseline moved refuses.

        Parameters
        ----------
        reference : dict
            The baseline's estimate and standard error.
        """
        self.reference = dict(reference)

    def _request(self) -> dict[str, Any] | None:
        assert self.checkpoint is not None
        path = Path(self.checkpoint.directory) / _REQUEST
        if not path.exists():
            return None
        stored: dict[str, Any] = json.loads(path.read_text())
        return stored

    def _has_draws(self) -> bool:
        assert self.checkpoint is not None
        return any(Path(self.checkpoint.directory).glob("*/draw-*.json"))

    def _write(self, request: dict[str, Any]) -> None:
        assert self.checkpoint is not None
        if self.read_only:
            return
        _write_json(Path(self.checkpoint.directory) / _REQUEST, request)

    def _check_common(self) -> None:
        assert self.checkpoint is not None
        stored = self._request()
        if stored is None or not self._has_draws():
            # A request with no stored draw holds nothing a new request could mix with, so a
            # call stopped before its first draw finished leaves the directory reusable.
            self._write({"common": self.parts, "tests": {}})
            return
        common = stored.get("common", {})
        for key, value in self.parts.items():
            if _digest(common.get(key)) == _digest(value):
                continue
            if key == "seeds" and common.get(key, {}).get("root") != value["root"]:
                resolved = None if self.drawn else value["root"]
                raise seed_refusal(self.checkpoint, int(common[key]["root"]), resolved)
            raise _refuse(
                self.checkpoint.directory,
                f"holds draws of another request: its {_PART_NAMES[key]} differs",
                "Use a fresh directory for this request.",
            )

    def test(self, name: str, parts: dict[str, Any]) -> None:
        """Check, or record, the parts of one stored test.

        Parameters
        ----------
        name : str
            The test.
        parts : dict
            Its draw count, process or declaration, bands, rule and resampling.
        """
        if self.checkpoint is None:
            return
        if self.reference is not None:
            parts = {**parts, "sampled baseline": self.reference}
        keyed = {key: config_key(value, path=f"{name}.{key}") for key, value in parts.items()}
        stored = self._request() or {"common": self.parts, "tests": {}}
        previous = stored.setdefault("tests", {}).get(name)
        if previous is None or not any(
            (Path(self.checkpoint.directory) / name).glob("draw-*.json")
        ):
            stored["tests"][name] = keyed
            self._write(stored)
            return
        if _digest(previous) != _digest(keyed):
            differing = sorted(
                key for key in set(previous) | set(keyed) if previous.get(key) != keyed.get(key)
            )
            raise _refuse(
                self.checkpoint.directory,
                f"holds draws of another {name} request: its {', '.join(differing)} differs",
                "Use a fresh directory for this request.",
            )

    def load(self, name: str, replicate: int, seed: int) -> Any:
        """The stored draw, checked against the seed this call derives, or ``None``.

        Parameters
        ----------
        name : str
            The test.
        replicate : int
            The draw's index.
        seed : int
            The seed this call derives for the draw.

        Returns
        -------
        EmpiricalRefitRecord, ReplicationFailure or None
            The stored draw.
        """
        if self.checkpoint is None:
            return None
        stored = self.checkpoint.load(name, replicate)
        if stored is None:
            return None
        if int(stored.seed) != int(seed):
            raise _refuse(
                self.checkpoint.directory,
                f"stores draw {replicate} of {name} at seed {stored.seed}, and this call derives "
                f"{seed}",
                "Use a fresh directory for this request.",
            )
        return stored

    def save(self, name: str, replicate: int, outcome: Any) -> None:
        """Store a finished draw.

        Parameters
        ----------
        name : str
            The test.
        replicate : int
            The draw's index.
        outcome : EmpiricalRefitRecord or ReplicationFailure
            The draw.
        """
        if self.checkpoint is not None and not self.read_only:
            self.checkpoint.save(name, replicate, outcome)

    def report(self, record: RefutationProgress) -> None:
        """Call the progress callback, if any, with ``record``.

        Parameters
        ----------
        record : RefutationProgress
            The test's progress.
        """
        if self.progress is not None:
            self.progress(record)
