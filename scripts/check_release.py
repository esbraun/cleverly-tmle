"""Validate a release tag against the package version."""

from __future__ import annotations

import re
import runpy
import sys
from pathlib import Path

ALPHA_TAG = re.compile(r"v(0\.1\.(?:0|[1-9][0-9]*))\Z")
VERSION_FILE = Path(__file__).parents[1] / "src" / "cleverly" / "_version.py"


def validate_tag(tag: str) -> str:
    """Return the version for a valid alpha tag that matches the source version."""
    match = ALPHA_TAG.fullmatch(tag)
    if match is None:
        raise ValueError(f"release tag {tag!r} must have the form v0.1.N")

    version = runpy.run_path(str(VERSION_FILE))["__version__"]
    if not isinstance(version, str):
        raise TypeError("__version__ must be a string")
    if match.group(1) != version:
        raise ValueError(f"release tag {tag!r} does not match package version {version!r}")
    return version


def main(argv: list[str] | None = None) -> None:
    """Validate the one tag supplied on the command line."""
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        raise SystemExit("usage: python scripts/check_release.py TAG")
    try:
        version = validate_tag(arguments[0])
    except (TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(version)


if __name__ == "__main__":
    main()
