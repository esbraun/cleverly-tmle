# E2R's frozen selection

`selection.json` goes here, and it is the one file in this repository that a study takes as an
**input** rather than producing as an output.

## What it is

`benchmarks/drtmle_reference_study.py --phase select` fits the control arm on the **selection**
cohort of draws, ranks the knot ladder for each of `qr`, `gr1` and `gr2`, and writes this file. It
carries four things, and each is checked rather than kept for the reader:

| part | what a decision run does with it |
| --- | --- |
| `rule` | the frozen constants at the selection. A decision run under a rule that moved since is **refused**, not re-judged |
| `configuration` | what the reference was built at — block sizes, ladder, control, learner, tier. A rung selected at one block size is a statement about that block size |
| `cohorts` | both draw sets, by seed. A decision run sharing a **data seed** with the selection cohort is refused |
| `selected` | the mapping itself, with the evidence that chose it |

`benchmarks.drtmle_reference_study.validate_selection` is where all four are read, and it runs
**before** the decision cohort is fitted — a run that could not have been certified must not be a
run that produced numbers first.

## Why it is committed

Because "the mapping was frozen before the deciding draws existed" is otherwise an honour system.
The selection is data-dependent: it is chosen across a set of draws, so certifying it needs a set of
draws it was not chosen on, and the commit is what makes the order checkable by someone who was not
in the room. [§8's decision protocol](../../docs/drtmle/validation-plan.md#the-decision-protocol-frozen-before-the-dispatch)
is the rule; this directory is where it lands.

It is therefore the exception to the rule that benchmark output is git-ignored, exactly as the
archived per-replicate rows beside it are, and for a related reason: a summary table is not the
evidence, and a mapping nobody can read is not a frozen mapping.

## The order

```
--phase select   ->  commit selection.json here  ->  --phase decide, once
```

The selecting dispatch reports no gates and no comparison; the deciding dispatch chooses nothing.
If a fidelity or integrity gate then fails, the cell is `unresolved` and the reduction-reference
branch stops — an `unresolved` E2R does not earn a third dispatch.
