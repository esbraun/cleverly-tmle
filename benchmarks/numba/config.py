"""What to run: sizes, core counts, repetitions, and the per-kernel dimension sweeps.

A config is a description of a *sweep*, not of one measurement.  The runner takes the
cross product of kernels, dimension settings, implementations and core counts, so the
config's job is to say which values each of those ranges over and to refuse a combination
that cannot mean anything -- a core count above what the process booted with, a dimension
a kernel does not take.

YAML is supported and optional.  ``pyyaml`` is not a dependency of this repository and
adding one for a benchmark would be the wrong trade, so a config file may be YAML *or*
JSON and the YAML path is taken only when the parser is importable.  Everything a config
file can say, the command line can say too.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

__all__ = ["Config", "KernelSweep", "load"]


@dataclass(frozen=True)
class KernelSweep:
    """One kernel's dimension settings.

    Each entry maps a dimension name to the *list of values* to sweep it over.  A scalar
    is accepted and treated as a one-element list, because writing ``n_arms: 2`` is the
    natural thing and failing on it would be pedantry.
    """

    kernel: str
    dimensions: Mapping[str, list[Any]] = field(default_factory=dict)

    def settings(self) -> list[dict[str, Any]]:
        """The cross product of the swept dimensions, as a list of keyword dicts."""
        names = sorted(self.dimensions)
        if not names:
            return [{}]
        out: list[dict[str, Any]] = [{}]
        for name in names:
            values = self.dimensions[name]
            out = [{**base, name: value} for base in out for value in values]
        return out


@dataclass(frozen=True)
class Config:
    """A whole run."""

    kernels: tuple[str, ...] = ("all",)
    scenarios: tuple[str, ...] = ()
    implementations: tuple[str, ...] = ("all",)
    sizes: tuple[int, ...] = (10_000, 100_000)
    num_cores: tuple[int, ...] = (1, 2, 4)
    repeats: int = 10
    warmups: int = 3
    #: Keep repeating past ``repeats`` until the measured calls total this long.  A kernel
    #: that runs in tens of microseconds needs it; one that runs in seconds does not.
    min_total_seconds: float = 0.5
    seed: int = 20260803
    output: Path = Path("benchmarks/results")
    validate: bool = True
    memory: bool = True
    cold_compile: bool = True
    amortise: bool = False
    #: Hybrid worker splits to try, as worker counts.  Empty means "the squarest split".
    hybrid_workers: tuple[int, ...] = ()
    sweeps: tuple[KernelSweep, ...] = ()

    def sweep_for(self, kernel: str) -> KernelSweep:
        for sweep in self.sweeps:
            if sweep.kernel == kernel:
                return sweep
        return KernelSweep(kernel=kernel)

    @property
    def max_cores(self) -> int:
        return max(self.num_cores) if self.num_cores else 1

    def with_output(self, output: Path) -> Config:
        return replace(self, output=output)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def load(path: str | Path) -> Config:
    """Read a config file.  YAML when the parser is available, JSON always."""
    text = Path(path).read_text()
    payload: Any
    if str(path).endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ImportError(
                f"{path} is YAML but pyyaml is not installed. Either `pip install pyyaml` "
                "or write the same content as JSON -- the loader accepts both, and pyyaml "
                "is deliberately not a dependency of this repository for a benchmark's sake"
            ) from error
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    return from_mapping(payload)


def from_mapping(payload: Mapping[str, Any]) -> Config:
    """Build a :class:`Config` from a parsed config document."""
    known = {
        "kernels",
        "scenarios",
        "implementations",
        "sizes",
        "num_cores",
        "repeats",
        "warmups",
        "min_total_seconds",
        "seed",
        "output",
        "validate",
        "memory",
        "cold_compile",
        "amortise",
        "hybrid_workers",
    }
    unknown = sorted(set(payload) - known - {"sweeps"})
    if unknown:
        raise KeyError(
            f"unknown config key(s) {unknown}. A typo here would otherwise run the "
            f"defaults and report them as though they were what was asked for; the "
            f"recognised keys are {sorted(known | {'sweeps'})}"
        )
    sweeps = tuple(
        KernelSweep(kernel=name, dimensions={k: _as_list(v) for k, v in dims.items()})
        for name, dims in (payload.get("sweeps") or {}).items()
    )
    fields: dict[str, Any] = {"sweeps": sweeps}
    for key in known:
        if key not in payload:
            continue
        value = payload[key]
        if key in ("kernels", "scenarios", "implementations"):
            fields[key] = tuple(_as_list(value))
        elif key in ("sizes", "num_cores", "hybrid_workers"):
            fields[key] = tuple(int(v) for v in _as_list(value))
        elif key == "output":
            fields[key] = Path(value)
        elif key in ("repeats", "warmups", "seed"):
            fields[key] = int(value)
        elif key == "min_total_seconds":
            fields[key] = float(value)
        else:
            fields[key] = bool(value)
    return Config(**fields)
