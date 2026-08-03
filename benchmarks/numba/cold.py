"""First-call compilation time, measured the only way it can be: in a fresh process.

Numba caches a compiled signature in memory for the life of the process, so "the first
call" is only the first call *once*.  Timing it inside a sweep that has already run the
kernel measures a dictionary lookup.  So each kernel's cold time is taken by launching a
subprocess that imports the module, builds a small fixture and calls the implementation
exactly once.

Small on purpose: compilation time depends on the *types* and the code, not on the array
lengths, so a 1,000-row fixture compiles the same kernel a 1,000,000-row one would and the
run time it also measures is then negligible beside the compile.  What is subtracted is the
warm time at the same size, so the reported number is the compilation rather than the
compilation plus one call.

The number that matters downstream is not the compile time itself but
:func:`~.timing.break_even_calls`: a four-second compile against a saving of 300 ms a call
pays for itself in fourteen calls, which a sensitivity sweep reaches and a single fit does
not.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config

__all__ = ["cold_compile_seconds", "report_cold_compile"]

_CHILD = """
import json, sys, time
from benchmarks.numba.kernels import resolve
spec = {spec!r}
name = {name!r}
[kernel] = [k for k in resolve([spec])]
inputs = kernel.inputs(**json.loads({dimensions!r}))
implementation = kernel.implementations[name]
start = time.perf_counter()
implementation(inputs)
first = time.perf_counter() - start
best = float("inf")
for _ in range(5):
    start = time.perf_counter()
    implementation(inputs)
    best = min(best, time.perf_counter() - start)
print(json.dumps({{"first": first, "warm": best}}))
"""


def cold_compile_seconds(kernel: str, implementation: str, dimensions: dict) -> dict[str, float]:
    """Launch a subprocess, call the kernel once, and report the compile time."""
    script = _CHILD.format(spec=kernel, name=implementation, dimensions=json.dumps(dimensions))
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if result.returncode != 0:
        return {"first": float("nan"), "warm": float("nan"), "compile": float("nan")}
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    payload["compile"] = max(0.0, payload["first"] - payload["warm"])
    return payload


def report_cold_compile(config: Config) -> int:
    """Print a compile-time table for every numba implementation of every kernel."""
    from .kernels import resolve
    from .timing import break_even_calls

    print(f"{'kernel':28} {'implementation':30} {'compile':>9} {'warm':>9} {'break-even':>11}")
    print("-" * 90)
    for spec in resolve(config.kernels):
        dimensions = {key: value for key, value in spec.dimensions.items() if key not in ("n",)}
        # A small fixture: compilation is a function of the types and the code, not the
        # length, and a large one would spend minutes running what it only needs to call.
        if "n" in spec.dimensions:
            dimensions["n"] = 2_000
        reference = None
        for name in spec.implementations:
            timings = cold_compile_seconds(spec.name, name, dimensions)
            if name == "numpy":
                reference = timings["warm"]
                continue
            if not name.startswith("numba"):
                continue
            saving = (reference - timings["warm"]) if reference else 0.0
            calls = break_even_calls(timings["compile"], saving)
            calls_text = f"{calls:.0f}" if calls == calls and calls != float("inf") else "never"
            print(
                f"{spec.name:28} {name:30} {timings['compile']:8.2f}s "
                f"{timings['warm'] * 1e3:8.2f}ms {calls_text:>11}"
            )
    print(
        "\nBreak-even is calls at *this* fixture size, so it is an upper bound: the saving "
        "grows with n and the compile does not."
    )
    return 0
