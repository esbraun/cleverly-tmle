#!/usr/bin/env bash
#
# Recover F4's contrast rows from a `--phase run` job log, and check them.
#
# `docs/drtmle/validation-plan.md` section 9 is the rule those rows are read against and
# `docs/drtmle/construction-contrasts.md` is the record they go into. Getting them out of a
# dispatch is the step the environment these studies are dispatched from cannot take the
# ordinary way: the rows leave the runner inside an Actions artefact, artefacts are served from
# `*.blob.core.windows.net`, and that host is refused at the sandbox's proxy with a
# `403 CONNECT tunnel failed`. `evidence/README.md` records the same measurement, which is why
# `scripts/fetch_evidence.sh` says to run it elsewhere.
#
# So the rows are printed into the job's log as well as uploaded, and this is the command that
# takes them back out. What makes that a recovery rather than a transcription is the digest: the
# log carries `CONTRASTS-SHA256` of the file the run wrote, this script decodes the gzipped block
# beside it and refuses unless the two agree. A hand-copied table is exactly what a record read
# against a frozen rule exists to rule out.
#
# The same command recovers the two other blocks a dispatch can print. `--label` selects which:
#
#     CONTRASTS   the paired contrast rows of a `--phase run` job          (default)
#     PREREG      the frozen manifest a `--phase prereg` run wrote
#     TRUNCATION  the exact reading of F4's sixth factor
#
# The log is whatever you saved the job's output to -- the Actions UI's "download log", the
# API's, or a tool that returns the content. Leading `2026-01-01T00:00:00.0000000Z ` timestamps
# are stripped when present and ignored when not.
#
# This does not validate the rows, only the bytes. `benchmarks.drtmle_construction.validate_prereg`
# is what checks the rule, the pinned configuration and the cohort disjointness, and `--phase run`
# runs it before it fits anything.
#
# Usage:
#     scripts/recover_construction.sh <job-log> [destination] [--label CONTRASTS|PREREG|TRUNCATION]
#
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    echo "usage: $(basename "$0") <job-log> [destination] [--label LABEL]" >&2
    echo >&2
    echo "  job-log      a saved drtmle-construction job log" >&2
    echo "  destination  where to write; defaults per label under benchmarks/results/" >&2
    echo "  --label      CONTRASTS (default), PREREG or TRUNCATION" >&2
    exit 2
}

LABEL="CONTRASTS"
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --label)
            [ $# -ge 2 ] || usage
            LABEL="$2"
            shift 2
            ;;
        -h | --help) usage ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

[ ${#ARGS[@]} -ge 1 ] || usage
readonly LOG="${ARGS[0]}"

case "$LABEL" in
    CONTRASTS) default="$ROOT/benchmarks/results/drtmle-construction/contrasts.jsonl" ;;
    PREREG) default="$ROOT/evidence/f4-construction/prereg.json" ;;
    TRUNCATION) default="$ROOT/benchmarks/results/drtmle-construction/truncation.jsonl" ;;
    *)
        echo "error: unknown label $LABEL; choose CONTRASTS, PREREG or TRUNCATION" >&2
        exit 2
        ;;
esac
readonly DEST="${ARGS[1]:-$default}"

[ -f "$LOG" ] || {
    echo "error: no log at $LOG" >&2
    exit 1
}

# The step's own script is echoed into the log above its output, so every pattern here is
# anchored: `echo "CONTRASTS-SHA256 ..."` in the echoed block does not match a line that *is* a
# digest, and an indented `echo "--- BEGIN ..."` does not match a marker.
strip_timestamps() {
    sed -E 's/^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z[[:space:]]//' "$LOG"
}

expected_sha="$(strip_timestamps | sed -n "s/^$LABEL-SHA256 \([0-9a-f]\{64\}\)\$/\1/p" | tail -n 1)"
expected_bytes="$(strip_timestamps | sed -n "s/^$LABEL-BYTES \([0-9][0-9]*\)\$/\1/p" | tail -n 1)"

[ -n "$expected_sha" ] || {
    echo "error: no $LABEL-SHA256 line in $LOG." >&2
    echo "A job that reached that block prints one; a job that did not reach one has nothing" >&2
    echo "to recover. Check the log for a validation refusal -- \`--phase run\` exits non-zero" >&2
    echo "before it fits anything if the committed manifest disagrees with the code." >&2
    exit 1
}

payload="$(strip_timestamps | awk -v label="$LABEL" '
    $0 == "--- BEGIN " label ".gz.base64 ---" { inside = 1; next }
    $0 == "--- END " label ".gz.base64 ---"   { inside = 0 }
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
    echo "The recovered bytes are not the bytes the run wrote. Do not read them." >&2
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
echo "That digest is of the file the run wrote. A contrast is read against the rule frozen in"
echo "docs/drtmle/validation-plan.md section 9, and an effect on the selection cohort alone is"
echo "not an effect: it has to reproduce on the audit cohort, which is a separate dispatch."
