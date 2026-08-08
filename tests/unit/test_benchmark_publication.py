r"""A benchmark table is refused before it is written, not annotated after.

``benchmarks/numba/reporting.py`` already carries the evidence a reader would need: every
row records the git sha, the CPU model, the interpreter, numpy, numba and the BLAS backend,
and ``load_rows`` says why -- "appending a row from another machine is exactly the mistake
this package refuses to make silently, so the evidence stays on every row."

**Not silent is not the same as not published.**  Six columns a reader has to diff by eye
sit under a summary table with a verdict at the bottom of it, and the verdict is what gets
quoted.  ``CLAUDE.md`` records what that costs, in the present tense: *every* committed
benchmark number here predates the timing harness's rotation, so a rerun is a different
instrument rather than a replication, and a small ratio -- ``1.02x`` against ``0.98x`` --
was never resolved by the numbers on record.  The provenance blocks said so, in prose, and
the numbers were compared across the change anyway.

So :func:`~benchmarks.numba.reporting.refuse_mixed_environment` fails closed, and this
module is what says it does.  ``--append`` is the path that reaches it: a full sweep is
hours and a single kernel is minutes, so running it kernel by kernel is the normal way to
work, and it is exactly the way a row from before a numpy upgrade ends up beside a row from
after one.

**This module deliberately does not import numba.**  ``tests/unit/test_numba_harness.py``
skips wholesale without it -- rightly, since it measures kernels -- and a publication gate
that could only be checked on a machine with the ``bench`` extra installed would be
unchecked in the fast tier, in CI, and in the sandbox this repository is developed in.
``reporting.py`` imports numba nowhere, so this runs everywhere.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # the benchmarks package is checked out, not installed
    sys.path.insert(0, str(ROOT))

from benchmarks.numba.reporting import (  # noqa: E402
    FINGERPRINT,
    MixedEnvironment,
    Row,
    fingerprint,
    merge,
    mixed_fields,
    refuse_mixed_environment,
    write_all,
)

#: A row with every required field filled in, so a test can vary the one field it is about.
#: Values chosen to look like a real run rather than to be minimal -- a fingerprint made of
#: empty strings would agree with itself for the wrong reason.
BASELINE: dict[str, Any] = {
    "scenario": "influence",
    "operation": "cluster_sums",
    "implementation": "numpy",
    "n": 100_000,
    "num_cores_requested": 1,
    "num_cores_effective": 1,
    "blas_threads": 1,
    "numba_threads": 1,
    "workers": 1,
    "repeat_count": 7,
    "warm_seconds": 0.0123,
    "warm_iqr_seconds": 0.0002,
    "warm_min_seconds": 0.0121,
    "warm_max_seconds": 0.0129,
    "cpu_seconds": 0.0125,
    "peak_rss_bytes": 1 << 26,
    "rss_delta_bytes": 1 << 20,
    "peak_alloc_bytes": 1 << 21,
    "correct": True,
    "max_abs_error": 0.0,
    "max_rel_error": 0.0,
    "git_sha": "0" * 40,
    "python_version": "3.12.7",
    "numpy_version": "2.1.3",
    "numba_version": None,
    "blas_backend": "openblas",
    "cpu_model": "a particular box",
}


def row(**overrides: Any) -> Row:
    return Row(**{**BASELINE, **overrides})


@dataclasses.dataclass(frozen=True)
class _Environment:
    """What ``summarise`` reads off an environment record.

    A stub rather than :func:`benchmarks.numba.resources.environment_record`, and not for
    convenience: ``resources.py`` imports the POSIX-only ``resource`` module at the top, so
    calling the real thing would make this module unimportable on Windows -- where this
    repository is in fact developed.  The gate under test lives in ``reporting.py``, which
    imports neither.
    """

    git_sha: str = "0" * 40
    git_dirty: bool = False
    python_version: str = "3.12.7"
    platform: str = "a particular box"
    cpu_model: str = "a particular box"
    physical_cores: int = 4
    logical_cores: int = 8
    numpy_version: str = "2.1.3"
    scipy_version: str = "1.14.1"
    numba_version: str | None = None
    llvmlite_version: str | None = None
    blas_backend: str = "openblas"
    blas_threading_layer: str = "pthreads"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def test_the_fingerprint_is_the_box_and_the_build_and_not_the_axes() -> None:
    """What the fingerprint must *not* contain is as load-bearing as what it does.

    A sweep varies ``n``, the core count and the implementation on purpose -- that variation
    is the table.  Putting any of them in the fingerprint would refuse every real run, which
    is the way a fail-closed gate gets turned off.
    """
    assert set(FINGERPRINT) == {
        "git_sha",
        "cpu_model",
        "python_version",
        "numpy_version",
        "numba_version",
        "blas_backend",
    }
    fields = {field.name for field in dataclasses.fields(Row)}
    assert set(FINGERPRINT) <= fields, "the fingerprint names a field Row does not have"
    for axis in ("n", "num_cores_requested", "implementation", "operation", "workers"):
        assert axis not in FINGERPRINT, f"{axis} is an axis of the sweep, not part of the box"


def test_one_box_is_not_mixed() -> None:
    rows = [row(n=n, implementation=impl) for n in (1_000, 100_000) for impl in ("numpy", "numba")]
    assert mixed_fields(rows) == {}
    assert len({fingerprint(one) for one in rows}) == 1
    refuse_mixed_environment(rows)  # does not raise


@pytest.mark.parametrize("field_name", FINGERPRINT)
def test_any_fingerprint_field_disagreeing_is_a_refusal(field_name: str) -> None:
    """Each field on its own, so a gate that only looked at the git sha would fail here."""
    other = "something else" if field_name != "numba_version" else "0.61.0"
    rows = [row(), row(**{field_name: other})]
    assert set(mixed_fields(rows)) == {field_name}
    with pytest.raises(MixedEnvironment, match="one box and one build"):
        refuse_mixed_environment(rows)


def test_a_skipped_row_is_not_a_mixture() -> None:
    """The false positive that would make the gate unusable on a machine without numba.

    A skipped row entered no plan and ran no code; it records the build it *would* have run
    on.  Counting it would make every sweep on a box without the ``bench`` extra refuse
    itself, which is the sandbox this repository is developed in.
    """
    rows = [row(), row(implementation="numba", skipped_reason="numba absent", numba_version="0.61")]
    assert mixed_fields(rows) == {}
    refuse_mixed_environment(rows)


def test_the_append_path_is_where_this_bites(tmp_path: Path) -> None:
    """The realistic failure, end to end: a partial re-run after a numpy upgrade.

    ``merge`` carries forward every row the second run did not touch, which is the whole
    point of ``--append`` -- and it is also how a row measured against numpy 2.1 ends up in
    one table with a row measured against 2.2, under a verdict that reads as a comparison
    of implementations.
    """
    before = [row(operation="cluster_sums"), row(operation="multiplier_bootstrap")]
    after = [row(operation="cluster_sums", numpy_version="2.2.0", warm_seconds=0.0090)]
    merged = merge(before, after)
    # Two, not three: ``after`` re-measured ``cluster_sums`` and so *replaces* it, which is
    # what --append is for. The row that was not re-run is carried forward at the old numpy
    # -- and that carried-forward row is the whole mixture, which is why this is the path
    # the gate exists on rather than a contrived one.
    assert len(merged) == 2
    assert {one.numpy_version for one in merged} == {"2.1.3", "2.2.0"}

    with pytest.raises(MixedEnvironment) as raised:
        write_all(merged, _Environment(), tmp_path)
    message = str(raised.value)
    assert "numpy_version" in message and "2.1.3" in message and "2.2.0" in message
    assert "Re-run the sweep" in message or "re-run" in message.lower()

    # Nothing on disk: a half-written results.jsonl is what the *next* --append reads, so
    # refusing after writing it would let the mixture survive the refusal.
    assert not (tmp_path / "latest" / "results.jsonl").exists()


def test_a_clean_set_still_writes(tmp_path: Path) -> None:
    """The other half of a fail-closed gate: it has to let the normal case through."""
    latest = write_all([row(), row(n=1_000)], _Environment(), tmp_path)
    assert (latest / "results.jsonl").exists()
    assert (latest / "summary.md").exists()
