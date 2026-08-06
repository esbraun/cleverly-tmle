# The evidence store

Where a dispatched study's **per-replicate rows** live, as opposed to the summary tables the
roadmap argues from. [The standing
decision](../docs/roadmap.md#standing-decisions) is the reason this directory exists: a CI
artefact has a ninety-day retention, and a summary table is a transcription of the evidence
rather than the evidence.

[`docs/drtmle/study-manifest.md`](../docs/drtmle/study-manifest.md) is the record in prose —
what was dispatched, at what code, with which inputs, and every artefact's digest.
[`manifest.json`](manifest.json) beside this file is the same record in a form
`scripts/fetch_evidence.sh` can read. `tests/unit/test_evidence_manifest.py` asserts the two
cover each other in both directions, so a digest corrected in one place and not the other is a
test failure rather than a silent disagreement.

## Fetching

```bash
scripts/fetch_evidence.sh e1b        # the eight companion-grid artefacts
scripts/fetch_evidence.sh c3c        # the four coverage-study artefacts
scripts/fetch_evidence.sh e2r        # the selecting run's, and the four deciding jobs'
```

**Not from the Claude Code sandbox.** Artefacts are served from `*.blob.core.windows.net`,
which it cannot reach — the request is refused at the proxy rather than by GitHub, so a fetch
there fails with a `403 CONNECT tunnel failed` that says nothing about whether the artefact
exists. Run it on a machine with ordinary outbound access.

**Before 2026-11-03.** After that the digests on record are a statement of what was measured
and no longer a way to obtain it.

## What is committed and what is not

The unpacked `.jsonl` is the point of this directory and belongs in the commit. The `.zip`
files are working files of the fetch and are git-ignored: they are the byte stream the digests
are of, so they are worth keeping until the check passes and worth nothing afterwards.

`sha256` in the manifest is the GitHub Actions API's digest of the artefact **zip**. It is not
the digest of the unpacked `.jsonl`, and the retrieval command this directory replaced hashed
the latter against the former — which is why `fetch_evidence.sh` exists rather than a line of
`gh run download`.

## Status

Nothing is fetched yet. This directory currently carries the manifest, the retrieval command
and this note, and the studies it names are still held only as CI artefacts under the
retention above. Until a fetch lands, `docs/drtmle/study-manifest.md` names evidence the
repository does not carry.
