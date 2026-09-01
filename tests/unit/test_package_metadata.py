"""Installed package metadata and release-version contracts."""

from __future__ import annotations

import re
from importlib.metadata import metadata, version

import pytest
from scripts.check_release import validate_tag

import cleverly


def test_runtime_version_matches_distribution_metadata() -> None:
    assert cleverly.__version__ == version("cleverly")
    assert re.fullmatch(r"0\.1\.(?:0|[1-9][0-9]*)", cleverly.__version__)


def test_distribution_metadata_describes_the_release() -> None:
    package = metadata("cleverly")
    assert package["Name"] == "cleverly"
    assert package["Requires-Python"] == ">=3.11"
    assert package["License-Expression"] == "MIT"
    assert package.get_all("License-File") == ["LICENSE"]
    classifiers = package.get_all("Classifier", [])
    assert "Development Status :: 3 - Alpha" in classifiers
    assert not any(classifier.startswith("License ::") for classifier in classifiers)
    assert any(
        url == "Issues, https://github.com/esbraun/cleverly-tmle/issues"
        for url in package.get_all("Project-URL", [])
    )


def test_release_tag_matches_the_source_version() -> None:
    assert validate_tag(f"v{cleverly.__version__}") == cleverly.__version__


@pytest.mark.parametrize("tag", ["0.1.0", "v0.2.0", "v0.1.01", "v0.1.0rc1"])
def test_release_tag_refuses_other_forms(tag: str) -> None:
    with pytest.raises(ValueError, match="must have the form"):
        validate_tag(tag)
