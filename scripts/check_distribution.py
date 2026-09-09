"""Check the contents of the wheel and source distribution."""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def _members(archive: Path) -> set[PurePosixPath]:
    """Return archive member paths relative to the distribution root."""
    if archive.suffix == ".whl":
        with zipfile.ZipFile(archive) as wheel:
            return {PurePosixPath(name) for name in wheel.namelist() if not name.endswith("/")}

    with tarfile.open(archive, mode="r:gz") as source:
        names = [PurePosixPath(member.name) for member in source.getmembers() if member.isfile()]
    roots = {name.parts[0] for name in names}
    if len(roots) != 1:
        raise AssertionError(f"{archive.name} does not have one source-distribution root")
    return {PurePosixPath(*name.parts[1:]) for name in names}


def _require(members: set[PurePosixPath], suffix: PurePosixPath, archive: Path) -> None:
    if not any(
        member == suffix or member.parts[-len(suffix.parts) :] == suffix.parts for member in members
    ):
        raise AssertionError(f"{archive.name} does not contain {suffix}")


def _forbid(members: set[PurePosixPath], prefix: PurePosixPath, archive: Path) -> None:
    if any(member.parts[: len(prefix.parts)] == prefix.parts for member in members):
        raise AssertionError(f"{archive.name} unexpectedly contains {prefix}")


def check_distribution(directory: Path) -> None:
    """Check the one wheel and one source archive in ``directory``."""
    wheels = list(directory.glob("cleverly-*.whl"))
    sources = list(directory.glob("cleverly-*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise AssertionError(
            f"expected one wheel and one source distribution, found {wheels!r} and {sources!r}"
        )

    wheel_members = _members(wheels[0])
    for required in (
        PurePosixPath("cleverly/_version.py"),
        PurePosixPath("cleverly/py.typed"),
        PurePosixPath("licenses/LICENSE"),
    ):
        _require(wheel_members, required, wheels[0])
    for forbidden in (PurePosixPath("tests"), PurePosixPath("docs")):
        _forbid(wheel_members, forbidden, wheels[0])

    source_members = _members(sources[0])
    for required in (
        PurePosixPath("LICENSE"),
        PurePosixPath("README.md"),
        PurePosixPath("pyproject.toml"),
        PurePosixPath("src/cleverly/_version.py"),
        PurePosixPath("src/cleverly/py.typed"),
    ):
        _require(source_members, required, sources[0])
    _forbid(source_members, PurePosixPath("tests"), sources[0])


def main(argv: list[str] | None = None) -> None:
    """Run the archive checks from the command line."""
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        raise SystemExit("usage: python scripts/check_distribution.py DIST_DIRECTORY")
    check_distribution(Path(arguments[0]))


if __name__ == "__main__":
    main()
