"""Contracts for the queue machine's Docker clean-up (``tests.canonical.docker_hygiene``)."""

from __future__ import annotations

from pathlib import Path

from tests.canonical.docker_hygiene import Container, Image, plan, referenced_images


def test_the_repository_names_the_reference_images_its_studies_use() -> None:
    """The scan finds tags the drivers pin, so a clean-up never removes a study's image."""
    referenced = referenced_images()
    assert "cleverly-drtmle-reference:538a3a2" in referenced
    assert "cleverly-ltmle-reference:1.3-0" in referenced
    assert all(tag.startswith("cleverly-") and ":" in tag for tag in referenced)


def test_the_scan_reads_python_and_json_under_tests(tmp_path: Path) -> None:
    (tmp_path / "tests" / "canonical" / "x").mkdir(parents=True)
    (tmp_path / "tests" / "canonical" / "x" / "study.py").write_text(
        'IMAGE = "cleverly-x-reference:1.0"\n', encoding="utf-8"
    )
    (tmp_path / "tests" / "canonical" / "x" / "manifest.json").write_text(
        '{"image": "cleverly-y:abc123"}', encoding="utf-8"
    )
    (tmp_path / "tests" / "canonical" / "x" / "run.py").write_text(
        'IMAGE = f"cleverly-built:{COMMIT}"\nOTHER = "cleverly-bare"\n', encoding="utf-8"
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "ignored.py").write_text('"cleverly-z:9"', encoding="utf-8")
    assert referenced_images(tmp_path) == {
        "cleverly-x-reference:1.0",
        "cleverly-y:abc123",
        "cleverly-built:*",
        "cleverly-bare:*",
    }


def test_a_repository_named_without_a_literal_tag_keeps_every_tag() -> None:
    images = [Image("i1", "cleverly-built:abc", False), Image("i2", "cleverly-pinned:old", False)]
    chosen = plan([], images, referenced={"cleverly-built:*", "cleverly-pinned:new"})
    assert chosen.kept_tags == ["cleverly-built:abc"]
    assert chosen.images == ["cleverly-pinned:old"]


def test_the_plan_removes_only_stale_cleverly_leftovers() -> None:
    """Running containers, named images, other projects' images and in-use tags stay."""
    containers = [
        Container("c-running", "cleverly-old:1", "running", True),
        Container("c-stopped", "cleverly-a:1", "exited", True),
        Container("c-orphan", "0123abcd", "created", False),
        Container("c-other", "postgres:16", "exited", True),
    ]
    images = [
        Image("i1", "cleverly-a:1", False),
        Image("i2", "cleverly-old:1", False),
        Image("i3", "cleverly-stale:2", False),
        Image("i4", "postgres:16", False),
        Image("i5", "other-project:latest", False),
        Image("i6", "", True),
    ]
    chosen = plan(containers, images, referenced={"cleverly-a:1"})
    assert chosen.containers == ["c-stopped", "c-orphan"]
    assert chosen.images == ["cleverly-stale:2", "i6"]
    assert sorted(chosen.kept_tags) == ["cleverly-a:1", "cleverly-old:1"]


def test_a_dangling_image_a_kept_container_uses_stays() -> None:
    containers = [Container("c-other", "i6", "exited", True)]
    chosen = plan(containers, [Image("i6", "", True)], referenced=set())
    assert chosen.containers == []
    assert chosen.images == []
