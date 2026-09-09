"""Execute and provenance-stamp a committed notebook artifact.

Running this needs a network connection, a ``python3`` kernel, and the time it takes to refit
every estimator the notebook fits. Documentation builds stay offline and render the committed
outputs.

The stamp itself is built by :mod:`tests.notebooks`, which also says which half of it the fast
suite asserts equal. Keeping the digests there lets ordinary verification avoid a kernel. The
executor's mutation controls still import :mod:`nbclient` to check its execution policy.

``--check`` re-executes without writing and reports each cell whose non-image output moved. It
compares stream, HTML, JSON, and other MIME payloads. The fast suite compares digests, and a digest
cannot see a library change that moves a published number while every cell keeps its bytes. That
is not hypothetical: one re-execution moved the TWINS estimate from ``-0.0651`` to ``-0.0639``
with the code cells untouched.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any

import nbformat
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

#: The wall clock a fit prints in its provenance stamp.  The prefix holds this mask to the
#: package-owned provenance line.  A date printed by an analysis remains result-bearing text.
WALL_CLOCK = re.compile(
    r"(?P<prefix>data [0-9a-f]+ \| folds [0-9a-f]+ \| cleverly [^|\n]+ \| )"
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)?"
)

#: :class:`~nbclient.NotebookClient` skips cells tagged ``skip-execution`` by default.  A
#: private tag that the preflight rejects forces every nonblank code cell through the kernel.
NEVER_SKIP_TAG = "__cleverly_never_skip__"


def _mask_wall_clock(value: Any) -> Any:
    """Mask package provenance clocks recursively and preserve other values."""
    if isinstance(value, str):
        return WALL_CLOCK.sub(r"\g<prefix><wall clock>", value)
    if isinstance(value, list):
        return [_mask_wall_clock(item) for item in value]
    if isinstance(value, dict):
        return {key: _mask_wall_clock(item) for key, item in value.items()}
    return value


def comparable_outputs(notebook: Any) -> dict[str, list[str]]:
    """Map each cell id to its canonical non-image outputs.

    Raster and vector images are left out because renderers can change their bytes without moving
    an estimate.  Stream text and every non-image MIME payload remain, including HTML and JSON
    tables that a documentation build can publish.
    """
    collected: dict[str, list[str]] = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        outputs = []
        for output in cell.get("outputs", ()):
            comparable: dict[str, Any] = {"output_type": output.get("output_type")}
            if "name" in output:
                comparable["name"] = output["name"]
            if "text" in output:
                comparable["text"] = output["text"]
            if "data" in output:
                comparable["data"] = {
                    mime: payload
                    for mime, payload in output["data"].items()
                    if not mime.startswith("image/")
                }
            for key in ("ename", "evalue", "traceback"):
                if key in output:
                    comparable[key] = output[key]
            outputs.append(
                json.dumps(
                    _mask_wall_clock(comparable),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        collected[cell["id"]] = outputs
    return collected


def validate_notebook_path(path: Path, *, repository_root: Path) -> Path:
    """Return a reader-facing notebook path, before any of its code can run."""
    resolved_root = repository_root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"notebook is outside the repository: {resolved}") from error
    if relative.suffix.casefold() != ".ipynb" or not resolved.is_file():
        raise ValueError(f"notebook is not a readable .ipynb file: {relative.as_posix()}")
    if not relative.parts or relative.parts[0] != "docs":
        raise ValueError(f"notebook is not reader-facing documentation: {relative.as_posix()}")
    excluded = {"_build", "generated", ".ipynb_checkpoints"}
    if excluded.intersection(relative.parts):
        raise ValueError(f"notebook is generated or autosaved content: {relative.as_posix()}")
    return resolved


def validate_execution_environment(*, repository_root: Path) -> None:
    """Refuse a kernel or editable install that points outside this checkout."""
    import cleverly

    expected = (repository_root / "src" / "cleverly").resolve()
    imported = Path(cleverly.__file__).resolve().parent
    if imported != expected:
        raise ValueError(
            f"import cleverly resolves to {imported}, not this checkout's source at {expected}"
        )
    executable = KernelSpecManager().get_kernel_spec("python3").argv[0]
    generic = {
        "python",
        f"python{sys.version_info.major}",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
    }
    same_executable = Path(executable).resolve() == Path(sys.executable).resolve()
    if executable not in generic and not same_executable:
        raise ValueError(
            f"the python3 kernelspec names {executable!r}, not the current interpreter"
        )


def execute_notebook(notebook: Any, *, timeout: int, repository_root: Path) -> None:
    """Execute every nonblank code cell with skipping and tolerated errors disabled."""
    blank = [
        cell.get("id")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and not str(cell.get("source", "")).strip()
    ]
    if blank:
        raise ValueError(f"blank code cell(s) cannot carry an execution artifact: {blank}")
    reserved = [
        cell.get("id")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and NEVER_SKIP_TAG in cell.get("metadata", {}).get("tags", ())
    ]
    if reserved:
        raise ValueError(
            f"code cell(s) use the executor's reserved tag {NEVER_SKIP_TAG!r}: {reserved}"
        )
    NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(repository_root)}},
        skip_cells_with_tag=NEVER_SKIP_TAG,
        force_raise_errors=True,
    ).execute()


def main() -> None:
    """Execute the notebook named on the command line, then write its stamp back.

    ``python scripts/execute_notebook.py <path>`` puts ``scripts/`` on the import path and not
    the repository root, so the stamp module is importable only after the first line here.
    That command is the one the stamp records and the one the contributor documentation gives,
    so the path repair belongs in the script rather than in an instruction to export
    ``PYTHONPATH``.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests.notebooks import REPOSITORY_ROOT, notebook_execution_stamp

    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-execute and report moved non-image output without writing the notebook",
    )
    args = parser.parse_args()

    try:
        path = validate_notebook_path(args.notebook, repository_root=REPOSITORY_ROOT)
        validate_execution_environment(repository_root=REPOSITORY_ROOT)
    except ValueError as error:
        parser.error(str(error))
    notebook = nbformat.read(path, as_version=4)
    stored = comparable_outputs(notebook)
    try:
        execute_notebook(notebook, timeout=args.timeout, repository_root=REPOSITORY_ROOT)
    except ValueError as error:
        parser.error(str(error))

    if args.check:
        fresh = comparable_outputs(notebook)
        moved = sorted(set(stored) | set(fresh), key=str)
        moved = [cell for cell in moved if stored.get(cell) != fresh.get(cell)]
        if moved:
            relative = path.relative_to(REPOSITORY_ROOT).as_posix()
            print(f"{relative} produced different non-image output in {moved}")
            for cell in moved:
                difference = difflib.unified_diff(
                    stored.get(cell, []),
                    fresh.get(cell, []),
                    fromfile=f"stored:{cell}",
                    tofile=f"fresh:{cell}",
                    lineterm="",
                )
                print("\n".join(difference))
            print("Read the difference before you regenerate. The stored artifact is what the")
            print("published prose interprets.")
            raise SystemExit(1)
        print(f"{path.name}: every cell reproduced its stored non-image output")
        return

    notebook.metadata["cleverly_execution"] = notebook_execution_stamp(notebook, path)
    nbformat.write(notebook, path)


if __name__ == "__main__":
    main()
