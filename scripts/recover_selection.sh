#!/usr/bin/env bash
#
# Recover E2R's frozen selection manifest from a `--phase select` job log, and check it.
#
# `docs/drtmle/validation-plan.md`'s decision protocol runs `--phase select`, has its manifest
# **committed**, and only then runs `--phase decide` at a mapping that run cannot change. Step 3
# is the commit, and it is the step the environment these studies are dispatched from cannot
# take: `selection.json` leaves the runner inside an Actions artefact, artefacts are served from
# `*.blob.core.windows.net`, and that host is refused at the sandbox's proxy with a
# `403 CONNECT tunnel failed`. `evidence/README.md` records the same measurement for the
# archived per-replicate rows, which is why `scripts/fetch_evidence.sh` says to run it elsewhere.
#
# **A freeze cannot be run elsewhere.** It sits between two dispatches of the same session, so
# the manifest is printed into the `select` job's log as well as uploaded, and this is the
# command that takes it back out. What makes that a recovery rather than a transcription is the
# digest: the log carries `SELECTION-SHA256` of the file the run wrote, this script decodes the
# gzipped block beside it and refuses unless the two agree. A hand-copied mapping is exactly
# what the commit exists to rule out.
#
# The log is whatever you saved the job's output to -- the Actions UI's "download log", the
# API's, or a tool that returns the content. Leading `2026-01-01T00:00:00.0000000Z ` timestamps
# are stripped when present and ignored when not.
#
# This does not validate the mapping, only the bytes.
# `benchmarks.drtmle_reference_study.validate_selection` is what checks the rule, the pinned
# configuration and the cohort disjointness, and `--phase decide` runs it before it fits
# anything.
#
# Usage:
#     scripts/recover_selection.sh <job-log> [destination]
#
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    echo "usage: $(basename "$0") <job-log> [destination]" >&2
    echo >&2
    echo "  job-log      a saved --phase select job log" >&2
    echo "  destination  where to write; defaults to evidence/e2r-selection/selection.json" >&2
    exit 2
}

[ $# -ge 1 ] || usage
readonly LOG="$1"
readonly DEST="${2:-$ROOT/evidence/e2r-selection/selection.json}"

[ -f "$LOG" ] || {
    echo "error: no log at $LOG" >&2
    exit 1
}

# The step's own script is echoed into the log above its output, so every pattern here is
# anchored: `echo "SELECTION-SHA256 ..."` in the echoed block does not match a line that *is* a
# digest, and an indented `echo "--- BEGIN ..."` does not match a marker.
strip_timestamps() {
    sed -E 's/^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z[[:space:]]//' "$LOG"
}

expected_sha="$(strip_timestamps | sed -n 's/^SELECTION-SHA256 \([0-9a-f]\{64\}\)$/\1/p' | tail -n 1)"
expected_bytes="$(strip_timestamps | sed -n 's/^SELECTION-BYTES \([0-9][0-9]*\)$/\1/p' | tail -n 1)"

[ -n "$expected_sha" ] || {
    echo "error: no SELECTION-SHA256 line in $LOG." >&2
    echo "A select job that reached its manifest prints one as its last step; a job that did" >&2
    echo "not reach one has nothing to freeze." >&2
    exit 1
}

payload="$(strip_timestamps | awk '
    /^--- BEGIN selection\.json\.gz\.base64 ---$/ { inside = 1; next }
    /^--- END selection\.json\.gz\.base64 ---$/   { inside = 0 }
    inside
')"

[ -n "$payload" ] || {
    echo "error: $LOG carries a digest and no base64 block; the log is truncated" >&2
    exit 1
}

readonly TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
printf '%s\n' "$payload" | tr -d ' \r' | base64 -d | gunzip >"$TMP"

actual_sha="$(sha256sum "$TMP" | cut -d' ' -f1)"
actual_bytes="$(wc -c <"$TMP" | tr -d ' ')"

if [ "$actual_sha" != "$expected_sha" ]; then
    echo "FAIL: recovered sha256 $actual_sha, log says $expected_sha" >&2
    echo "The recovered bytes are not the bytes the run wrote. Do not commit them." >&2
    exit 1
fi
if [ -n "$expected_bytes" ] && [ "$actual_bytes" != "$expected_bytes" ]; then
    echo "FAIL: recovered $actual_bytes bytes, log says $expected_bytes" >&2
    exit 1
fi

mkdir -p "$(dirname "$DEST")"
cp "$TMP" "$DEST"
echo "ok  $DEST  $actual_bytes bytes  $actual_sha"
echo
echo "That digest is of the file the select run wrote. Commit it before dispatching"
echo "--phase decide; the decision run validates the mapping itself and refuses one whose"
echo "draws are its own."
