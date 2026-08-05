#!/usr/bin/env bash
#
# Fetch a dispatched study's per-replicate rows and check them against the manifest.
#
# `docs/roadmap.md`'s standing decisions carry one that says a dispatched *study*'s rows are
# "archived rather than left in CI", because a CI artefact has a ninety-day retention and a
# summary table is a transcription of the evidence rather than the evidence. This is the
# command that discharges it. `evidence/manifest.json` is what it reads, and
# `docs/drtmle/study-manifest.md` is the same record in prose --
# `tests/unit/test_evidence_manifest.py` is what stops the two drifting.
#
# **`gh run download` is the wrong tool and that is why this exists.** It extracts, so it
# leaves no archive to hash, while every digest on record is the API's digest of the artefact
# *zip*. The check that used to be documented therefore compared a recorded number against a
# file nothing records the digest of, and would have passed or failed for reasons unrelated to
# the payload. This fetches the zip endpoint, which is the byte stream those digests are of,
# verifies, and only then unpacks.
#
# Needs a machine with ordinary outbound access: artefacts are served from
# `*.blob.core.windows.net`, which the sandbox these studies were dispatched from cannot reach
# -- the request is refused at the proxy rather than by GitHub. Run it before the retention
# date in the manifest, after which the digests are a record of what was measured and no
# longer a way to obtain it.
#
# Usage:
#     scripts/fetch_evidence.sh e1b [destination]
#     scripts/fetch_evidence.sh c3c [destination]
#     scripts/fetch_evidence.sh e2  [destination]
#
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly MANIFEST="$ROOT/evidence/manifest.json"

usage() {
    echo "usage: $(basename "$0") <study> [destination]" >&2
    echo >&2
    echo "  study        a key of evidence/manifest.json's \"studies\" -- e.g. e1b, c3c, e2" >&2
    echo "  destination  where to write; defaults to evidence/<study>" >&2
    exit 2
}

[ $# -ge 1 ] || usage
readonly STUDY="$1"
readonly DEST="${2:-$ROOT/evidence/$STUDY}"

for tool in gh jq sha256sum; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "error: $tool is required and is not on PATH" >&2
        exit 1
    }
done

[ -f "$MANIFEST" ] || {
    echo "error: no manifest at $MANIFEST" >&2
    exit 1
}

REPO="$(jq -r '.repo' "$MANIFEST")"
EXPIRES="$(jq -r '.retention_expires' "$MANIFEST")"

jq -e --arg s "$STUDY" '.studies[$s]' "$MANIFEST" >/dev/null 2>&1 || {
    echo "error: no study \"$STUDY\" in $MANIFEST; known studies:" >&2
    jq -r '.studies | keys[] | "  " + .' "$MANIFEST" >&2
    exit 1
}

# The expiry is a warning and not a refusal: a fetch after it will fail at the API with a
# message of its own, and a script that pre-empted that would be guessing at GitHub's
# retention behaviour rather than reporting it.
if [ "$(date -u +%Y-%m-%d)" \> "$EXPIRES" ]; then
    echo "warning: retention expired on $EXPIRES; these artefacts are very likely gone" >&2
fi

mkdir -p "$DEST"
echo "fetching $STUDY from $REPO into $DEST (retention expires $EXPIRES)"

failures=0
count=0
while IFS=$'\t' read -r artefact sha256 bytes; do
    count=$((count + 1))
    zip="$DEST/$artefact.zip"

    # The zip endpoint rather than `gh run download`: this is the byte stream the manifest's
    # digest is of. `gh api` follows the redirect to blob storage for us.
    gh api "/repos/$REPO/actions/artifacts/$artefact/zip" >"$zip"

    actual="$(sha256sum "$zip" | cut -d' ' -f1)"
    size="$(wc -c <"$zip" | tr -d ' ')"
    if [ "$actual" != "$sha256" ]; then
        echo "  FAIL $artefact: sha256 $actual, manifest says $sha256" >&2
        failures=$((failures + 1))
        continue
    fi
    if [ "$size" != "$bytes" ]; then
        echo "  FAIL $artefact: $size bytes, manifest says $bytes" >&2
        failures=$((failures + 1))
        continue
    fi

    unzip -o -q "$zip" -d "$DEST/$artefact"
    echo "  ok   $artefact  $size bytes  $sha256"
done < <(jq -r --arg s "$STUDY" \
    '.studies[$s].artefacts[] | [.artefact, .sha256, .bytes] | @tsv' "$MANIFEST")

echo
if [ "$failures" -ne 0 ]; then
    echo "$failures of $count artefacts did not match the manifest. The payload is not the" >&2
    echo "record until they do; do not commit it." >&2
    exit 1
fi
echo "$count of $count artefacts match the manifest."
echo "Commit the unpacked .jsonl; the .zip files are git-ignored."
