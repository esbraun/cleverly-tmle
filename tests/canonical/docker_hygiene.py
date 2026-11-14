"""Remove the Docker leftovers that study runs accumulate on a queue machine.

Run it after each local-queue job, and weekly::

    python -m tests.canonical.docker_hygiene            # print what it would remove
    python -m tests.canonical.docker_hygiene --apply    # remove it

The drivers already run their containers with ``--rm``, so a finished run leaves no container.
What builds up is everything else: a container a killed run left behind, the untagged image each
rebuild of a reference image leaves, ``cleverly-*`` reference images whose tag no study still
names, and BuildKit's build cache.  This script removes exactly those:

- stopped containers whose image is a ``cleverly-*`` image or is gone;
- dangling (untagged, unreferenced) images;
- ``cleverly-*`` tags that no file under ``tests/`` names at the checked-out commit, unless a
  container uses them (a repository named without a literal tag keeps all its tags);
- build cache unused for longer than ``--cache-age`` (default one week).

It never removes a volume (``cleverly-cache`` holds the R runs' checkpoints), a running
container, an image a study still names, or an image of another project.  Run it from a
checkout of ``main`` so that the referenced tags are the current ones; a branch that pins a
newer reference image keeps its tag only while that branch is checked out.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: A ``cleverly-*`` name and, when it is written out, its literal tag.
IMAGE_NAME = re.compile(
    r"\b(cleverly-[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?)(?::([A-Za-z0-9][A-Za-z0-9._-]*))?"
)

#: The files that name a study's image.
SOURCE_GLOBS = ("**/*.py", "**/*.json", "**/*.md", "**/Dockerfile")


@dataclass(frozen=True)
class Container:
    """One container as ``docker ps -a`` lists it.

    Parameters
    ----------
    identifier : str
        The container ID.
    image : str
        The image the container was created from, as a tag or an image ID.
    state : str
        Docker's state: ``running``, ``exited``, ``created`` and so on.
    image_exists : bool
        Whether Docker still holds that image.
    """

    identifier: str
    image: str
    state: str
    image_exists: bool


@dataclass(frozen=True)
class Image:
    """One image tag (or one untagged image) as ``docker images`` lists it.

    Parameters
    ----------
    identifier : str
        The image ID.
    reference : str
        ``repository:tag``, or ``""`` for an untagged image.
    dangling : bool
        Whether the image has no tag and no child image.
    """

    identifier: str
    reference: str
    dangling: bool


@dataclass
class Plan:
    """What :func:`plan` would remove.

    Parameters
    ----------
    containers : list of str
        Container IDs to remove.
    images : list of str
        Image tags, or the IDs of dangling images, to remove.
    kept_tags : list of str
        ``cleverly-*`` tags kept because a study names them or a container uses them.
    """

    containers: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    kept_tags: list[str] = field(default_factory=list)


def referenced_images(root: Path = ROOT) -> set[str]:
    """Every ``cleverly-*`` image a file under ``root / "tests"`` names.

    A name written with a literal tag (``cleverly-x:1.0``) keeps that tag.  A name written
    without one (``cleverly-econml-parity``, or ``f"cleverly-zepid-reference:{commit}"``) keeps
    every tag of that repository, recorded as ``repository:*``, because the tag is built at run
    time or is the implicit ``latest``.

    Parameters
    ----------
    root : Path
        The repository root.

    Returns
    -------
    set of str
        ``repository:tag`` for each literal tag and ``repository:*`` for each open name.
    """
    found: set[str] = set()
    tests = root / "tests"
    for pattern in SOURCE_GLOBS:
        for path in tests.glob(pattern):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for repository, tag in IMAGE_NAME.findall(text):
                found.add(f"{repository}:{tag or '*'}")
    return found


def _named(reference: str, referenced: set[str]) -> bool:
    repository = reference.rsplit(":", 1)[0]
    return reference in referenced or f"{repository}:*" in referenced


def plan(containers: Iterable[Container], images: Iterable[Image], referenced: set[str]) -> Plan:
    """Choose the stale containers and images; touch nothing.

    Parameters
    ----------
    containers : iterable of Container
        Every container Docker holds.
    images : iterable of Image
        Every image Docker holds.
    referenced : set of str
        The ``cleverly-*`` tags a study names (:func:`referenced_images`).

    Returns
    -------
    Plan
        The containers and images to remove, and the ``cleverly-*`` tags kept.
    """
    result = Plan()
    containers = list(containers)
    for container in containers:
        if container.state == "running":
            continue
        if container.image.startswith("cleverly-") or not container.image_exists:
            result.containers.append(container.identifier)
    removed = set(result.containers)
    in_use = {c.image for c in containers if c.identifier not in removed}
    for image in images:
        if image.dangling:
            if image.identifier not in in_use:
                result.images.append(image.identifier)
            continue
        if not image.reference.startswith("cleverly-"):
            continue
        if _named(image.reference, referenced) or image.reference in in_use:
            result.kept_tags.append(image.reference)
        else:
            result.images.append(image.reference)
    return result


def _docker(*arguments: str) -> str:
    completed = subprocess.run(["docker", *arguments], capture_output=True, text=True, check=True)
    return completed.stdout


def _json_lines(text: str) -> list[dict[str, str]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def inventory() -> tuple[list[Container], list[Image]]:
    """Read Docker's containers and images.

    Returns
    -------
    tuple of (list of Container, list of Image)
        Every container and every image.
    """
    image_rows = _json_lines(_docker("images", "-a", "--no-trunc", "--format", "{{json .}}"))
    dangling_ids = set(_docker("images", "-q", "--no-trunc", "--filter", "dangling=true").split())
    known = {row["ID"] for row in image_rows}
    known |= {row["ID"].removeprefix("sha256:")[:12] for row in image_rows}
    images = [
        Image(
            identifier=row["ID"],
            reference=""
            if "<none>" in (row["Repository"], row["Tag"])
            else f"{row['Repository']}:{row['Tag']}",
            dangling=row["ID"] in dangling_ids,
        )
        for row in image_rows
    ]
    tags = {image.reference for image in images if image.reference}
    containers = []
    for row in _json_lines(_docker("ps", "-a", "--no-trunc", "--format", "{{json .}}")):
        name = row["Image"]
        exists = name in tags or name in known or name.removeprefix("sha256:") in known
        containers.append(
            Container(identifier=row["ID"], image=name, state=row["State"], image_exists=exists)
        )
    return containers, images


def main(argv: Sequence[str] | None = None) -> int:
    """Print, and with ``--apply`` carry out, the removal plan.

    Parameters
    ----------
    argv : sequence of str, optional
        The command-line arguments; ``sys.argv`` when omitted.

    Returns
    -------
    int
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="remove what the plan lists")
    parser.add_argument(
        "--cache-age",
        default="168h",
        help="prune build cache unused for longer than this (a Docker duration; default 168h)",
    )
    args = parser.parse_args(argv)
    containers, images = inventory()
    chosen = plan(containers, images, referenced_images())
    print(f"containers to remove: {len(chosen.containers)}")
    for identifier in chosen.containers:
        print(f"  {identifier[:12]}")
    print(f"images to remove: {len(chosen.images)}")
    for reference in chosen.images:
        print(f"  {reference.removeprefix('sha256:')[:40]}")
    print(f"cleverly images kept: {', '.join(sorted(chosen.kept_tags)) or 'none'}")
    print(f"build cache: prune entries unused for {args.cache_age}")
    if not args.apply:
        print("dry run: pass --apply to remove")
        return 0
    if chosen.containers:
        _docker("rm", *chosen.containers)
    if chosen.images:
        subprocess.run(["docker", "rmi", *chosen.images], check=False)
    subprocess.run(
        ["docker", "builder", "prune", "-f", "--filter", f"until={args.cache_age}"], check=True
    )
    print(_docker("system", "df"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
