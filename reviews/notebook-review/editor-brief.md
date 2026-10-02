# Notebook editor brief (changes N1 to N10)

You edit one tutorial notebook and its callback, then commit. Editors run one at a time.

## Setting

- Worktree, and the only place you work:
  `C:/Users/erics/Documents/Projects/cleverly-tmle/.claude/worktrees/bridge-cse_01BbvRGwt6wTxoguDudAmLTi`,
  branch `agent/notebook-review`. Windows, Git Bash. Python is the worktree's
  `.venv/Scripts/python.exe`. Assert that `cleverly.__file__` is under the worktree's `src`.
- Do not push. Do not edit `src/`, `tests/studies/`, `tests/canonical/`, or another notebook. If
  the notebook needs a library change, stop and report it.
- Never pipe a command whose exit status matters into `tail` or `head`. Redirect it to a file and
  echo `$?`. Do not use `jupyter nbconvert`.
- Use the Write or Edit tools, or Python with `json`, for notebook edits. Keep every cell id.
  Keep LF endings.
- Probes and seed sweeps may use up to 12 worker processes, each single-threaded
  (`OMP_NUM_THREADS=1`, `n_jobs=1`). Put sweep scripts and their outputs that the page cites
  under `reviews/notebook-review/probes/<stem>-final/`, and commit them. Scratch goes to
  `.tmp/notebook-review/edit-<stem>/`.

## Read first

1. `CLAUDE.md`, and `docs/development/example-notebooks.md` as changed by this branch. It now
   says that notebooks are working end-to-end examples that never showcase refusals. It also
   adds the sweep rules, the trust-row checklist, and the sensitivity table.
2. `reviews/notebook-review/plan.md`: the rulings R1 to R8, the decisions table, the declared
   study reruns with "What the run found", and your row.
3. Your rows in `reviews/notebook-review/ledger.md`, your section in `refusal-inventory.md`, and
   the relevant patterns in `sibling-sweep.md`. Also the findings and verification files for your
   stem, and the gates under `reviews/notebook-review/gates/`.
4. The library changes already on this branch:
   - S1 (5bc92b5e) bounds the `nonlinear_dgp` propensity to [0.05, 0.95]. `navigation_data` uses
     that law. The ATE is unchanged, and ATT and ATC moved.
   - S2 (d0fc2e45 and its fix) changes some printed strings. "maximal bias" is now "bias scale",
     the RV sentence changed, and the DR-TMLE contract line now names cross-fitting.
   - S3 (1f9cf969) regenerates four studies. Their numbers changed; see `plan.md`.

## Procedure

1. Plan the edits from the ledger rows, and the refusal replacement for each refusal cell.
2. Run any sweep the readings need before you write them (rules R3, R4, R8). Every comparative or
   property sentence must hold over at least 60 draws, or say "on this draw". Recompute every
   printed truth independently once, from the structural equations.
3. Edit the code cells. Use explicit, cheap learners, `n_jobs=1`, and every seed set. Keep the
   whole notebook under 60 s.
4. Run `ruff format` and `ruff check` on the notebook. Execute it once:
   `.venv/Scripts/python.exe scripts/execute_notebook.py docs/examples/<stem>.ipynb > .tmp/notebook-review/edit-<stem>/execute.log 2>&1; echo $?`.
   Then confirm that the execution counts are contiguous and that no output is an error.
5. Read every stored output. Then write each reading from those outputs only.
6. Update the callback in `tests/unit/tutorial_semantics/<module>.py`. Assert each seeded relation
   that the readings narrate, and give each nonzero witness. List every quoted unprinted decimal
   in `UNPRINTED_DECIMALS` with its source. Fix any stale comment.
7. Run `--check` (it writes nothing) and the targeted tests:
   - `scripts/execute_notebook.py <path> --check`
   - `pytest -q -p no:cacheprovider tests/unit/test_documentation_runtime.py -k <module>`
   - `pytest -q -p no:cacheprovider tests/unit/test_documentation_links.py -k <module>`
   - `python -m tests.prose --path docs/examples/<stem>.ipynb`. Do not run `--update`; the
     orchestrator refreshes the ledger at the end. Report each finding you keep, with a proposed
     `accepted: <reason>`.
   - Check that the notebook has no CRLF:
     `python -c "import sys; print(bytes([13,10]) in open(sys.argv[1],'rb').read())" <path>`
8. Commit the notebook, the callback, and the `probes/<stem>-final/` files. Use a house-style
   subject and a body that answers the four questions in `docs/development/pull-requests.md`. List
   the ledger IDs fixed, and any ID left unfixed with its reason. End with the gate line and
   `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Report

Write under 300 words, with:

- the commit hash
- the files changed
- the execution time and each command with its exit code
- a table with one row per ledger ID: fixed, or not fixed with the reason
- the sweep results the page quotes
- any finding you disagree with, and your evidence
- the prose findings you keep, with a proposed reason for each
