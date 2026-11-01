"""The registry of the X33 fixtures: one R script and one JSON file each.

Each fixture holds the R side of a comparison for a longitudinal fitting option:
``lmtp``'s ratio trim (part d, ``ratio_trim=``), ``lmtp``'s Markov history windows (part e,
``markov_order=``) and ``survtmle``'s pooled censoring regression (part j,
``hazard_pooling="pooled"``).  The entries reuse :class:`FixtureEntry` and
:func:`check_entry` of the targeting-option fixtures
(:mod:`tests.canonical.option_fixtures.registry`), so a pending fixture has no JSON file, a
reason and the checksum of its script, and a present one records that checksum and its image.
``tests/unit/test_longitudinal_option_fixtures.py`` asserts the state and runs the comparisons
(:mod:`tests.canonical.longitudinal_option_fixtures.parity`).

A missing comparator never blocks a method (``docs/architecture-invariants.md``), and the
Python exact tests of each option gate it.  The pending state is reported as XFAIL with its
reason, never read as a pass.
"""

from __future__ import annotations

from pathlib import Path

from tests.canonical.option_fixtures.registry import FixtureEntry, check_entry, script_sha256

__all__ = ["HERE", "REGISTRY", "build_command", "check_entry", "run_command", "script_sha256"]

HERE = Path(__file__).resolve().parent

REGISTRY: tuple[FixtureEntry, ...] = (
    FixtureEntry(
        name="lmtp_trim",
        parts=("d",),
        comparator="lmtp:::trim on a fixed matrix, and lmtp_tmle(control = lmtp_control(.trim))",
        script="lmtp_trim.R",
        image="cleverly-lmtp-crossfit:1.5.4",
        context="tests/canonical/lmtp_crossfit",
        packages=("lmtp",),
        status="present",
        reason="",
        script_sha256="ff38fde3a0a584134d2171523ce4b0972743695ebd40acd01ce26ee8adf3dee7",
    ),
    FixtureEntry(
        name="lmtp_history",
        parts=("e",),
        comparator="lmtp:::LmtpVars$new(..., k)$history(var, t) for k in 0, 1, 2 and Inf",
        script="lmtp_history.R",
        image="cleverly-lmtp-crossfit:1.5.4",
        context="tests/canonical/lmtp_crossfit",
        packages=("lmtp",),
        status="present",
        reason="",
        script_sha256="95c9369c0e18958f6a6f9253be12c03b58ea5e0011517955c74718edb1c91862",
    ),
    FixtureEntry(
        name="survtmle_pooled",
        parts=("j",),
        comparator="survtmle(method = 'mean') with per-visit and linear-t glm.ctime",
        script="survtmle_pooled.R",
        image="cleverly-survtmle:1.1.1",
        context="tests/canonical/survtmle",
        packages=("survtmle",),
        status="present",
        reason="",
        script_sha256="0d9adbe8920a8c17bf48eea13077e6c5c1b22affa8c55a5d05d9944b5ed2ed5f",
    ),
)


def build_command(entry: FixtureEntry) -> str:
    """The command that builds the pinned image, run from the repository root."""
    return f"docker build -t {entry.image} {entry.context}"


def run_command(entry: FixtureEntry) -> str:
    """The command the local Study queue runs for one fixture, from the repository root.

    The container mounts this directory and writes the JSON file beside the script.  The image
    id and the script checksum enter the provenance of the file.
    """
    mount = "$PWD/tests/canonical/longitudinal_option_fixtures"
    return (
        "docker run --rm --entrypoint Rscript "
        f"-e CLEVERLY_IMAGE_ID=\"$(docker image inspect --format '{{{{.Id}}}}' {entry.image})\" "
        f'-e CLEVERLY_SCRIPT_SHA256="$(sha256sum {mount}/{entry.script} | cut -d" " -f1)" '
        f'-v "{mount}:/fixture" {entry.image} /fixture/{entry.script} /fixture/{entry.output}'
    )


if __name__ == "__main__":
    for item in REGISTRY:
        print(f"# {item.name} {script_sha256(HERE / item.script)}")
        print(build_command(item))
        print(run_command(item))
