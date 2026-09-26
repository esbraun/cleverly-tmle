# RM18 density-representation attribution

This declaration fixes the RM18 comparator-density diagnostic before any diagnostic estimate is
computed. It redraws the 800 primary samples with `canonical_shift_policies.draw_scenario` at
the registered size and replicate indices. It reads the committed replication rows in
`tests/canonical/lmtp_shift/` as anchors. That directory does not commit sample frames. The
registered seed stream and the anchor checks below establish the replay. This diagnostic changes
no registered study, margin, artifact, or verdict.

## Four cells on each sample

| cell | engine | density ratio | construction |
| --- | --- | --- | --- |
| C-H | `cleverly` | pooled-hazard | reproduce the registered three-policy fit |
| C-A | `cleverly` | analytic | retarget C-H's cached nuisances after replacing both `ShiftSet.ratio` and `ratio_at` for all three policies |
| R-A | pinned R `lmtp` | analytic | reproduce the registered natural-course and `+0.25` fits |
| R-H | pinned R `lmtp` | pooled-hazard | pass C-H's observed-dose `+0.25` ratio to `lmtp_point_tmle`; share R-A's natural-course fit |

The analytic ratio uses the conditional-normal law in `canonical_shift_policies.py`. For an
uncapped shift at dose $a$, it is $g(a-\delta\mid W)/g(a\mid W)$. For a capped shift, it uses the
two preimage terms in `src/cleverly/interventions/shift.py`. The natural-course ratio is one.
Each C-A `ratio_at[i,s,r]` evaluates policy $r$ at policy $s$'s shifted dose for row $i$.
The R-H adapter receives the C-H ratio on each observed row. It does not estimate or transform
that ratio again. Both R quarter-shift fits use the same data, outcome learner, and adapter.

The diagnostic reports only `ate_shift[+0.25 vs natural course]`. It retains all three policies in
the C-H fit so its targeting state matches the registered study. The natural-course R fit is
shared between R-A and R-H. Each cell reports the estimate, standard error, and initial estimate.
The runner joins every cell to the same replicate and truth.

## Anchor and sample checks

Before calculating a reading, require exactly 800 distinct replicate keys, 2,000 rows per redrawn
sample, one truth per key, and four finite estimate and standard-error pairs per key. Reject a
missing, duplicate, or unmatched row. Compare C-H and R-A with their committed primary rows on
each key. The registered draw uses `SeedSequence` with the study seed, scenario index, and
replicate index. It does not depend on the total replication count.
Require the estimate, standard error, and initial estimate to differ by at most $10^{-6}$, and
require the truth, sample size, and coverage flag to agree exactly. Stop without a reading if
either anchor fails. The comparison checks that each redrawn sample reproduces its committed
outputs; no committed frame exists for a bytewise sample comparison.

## Calibration statistics and interval rule

For each cell, calculate its signed calibration error as the mean reported standard error divided
by the sample standard deviation of its 800 estimates, minus one. Denote its absolute value by
$a_{E,S}$, where $E$ is C or R and $S$ is H or A. Positive contrasts below mean the first cell
has a larger absolute calibration error.

| contrast | formula | question |
| --- | --- | --- |
| observed gap | $a_{C,H}-a_{R,A}$ | does the registered pair differ? |
| C density | $a_{C,H}-a_{C,A}$ | does the ratio source move `cleverly`? |
| R density | $a_{R,H}-a_{R,A}$ | does the ratio source move `lmtp`? |
| analytic engine | $a_{C,A}-a_{R,A}$ | what engine gap remains at the analytic ratio? |
| hazard engine | $a_{C,H}-a_{R,H}$ | what engine gap remains at the pooled-hazard ratio? |
| interaction | $(a_{C,H}-a_{C,A})-(a_{R,H}-a_{R,A})$ | does the ratio effect differ by engine? |

The observed gap has two exact additive decompositions:
$G=D_C+E_A=E_H+D_R$, where $D$ denotes a density contrast and $E$ an engine contrast.
The interaction is $D_C-D_R=E_H-E_A$. Both paths therefore constrain an attribution.

Resample the 800 paired replicate keys 20,000 times with replacement using NumPy
`default_rng(20260926)`. Recompute the four signed calibration errors within every sample.
Use the 0.00125 and 0.99875 quantiles for each error. These four marginal 99.75% intervals form a
simultaneous rectangular region with at least 99% nominal coverage by the Bonferroni rule.
Map each signed interval through absolute value. If it crosses zero, its mapped lower bound is
zero. Use interval arithmetic on these four mapped intervals to bound all six contrasts above.
This mapping avoids a direct percentile interval for an absolute error near zero.

Use $\epsilon=0.01$ in standard-error-ratio units as the predeclared practical equivalence margin.
It is one fifth of the registered 0.05 calibration non-inferiority margin.
Read **density sufficient** if the observed-gap lower bound is positive and either additive path
has a positive density lower bound and an equivalent same-ratio engine interval. Also require the
interaction interval to lie wholly inside $[-\epsilon,\epsilon]$. An interval is equivalent when
both its bounds lie inside this margin. Read **engine residual** if the observed-gap lower bound is
positive and either same-ratio engine interval lies wholly outside $[-\epsilon,\epsilon]$.
Read **mixed** if an engine-residual condition holds and either density lower bound is positive.
Give `mixed` priority over `engine residual`. Otherwise read **unresolved**. A density contrast
can show an effect without meeting the stronger `density sufficient` rule. The reading describes
finite-sample calibration in this law. It makes no universal causal attribution.

Write four-cell replication rows and the mapped intervals to new files in an explicit output
directory. Never overwrite the registered artifacts. The runner must verify both anchors before
it writes a reading. Commit and push this declaration before any diagnostic fit or result read.

## Cost and execution boundary

The full replay needs 800 `cleverly` nuisance fits, 800 cached-nuisance retargets, and at most
2,400 R policy fits: one natural-course fit and two quarter-shift fits per replicate. It then
performs 20,000 bootstrap draws of the 800 paired keys. A small code-path test may validate array
shape and ratio routing before the full replay. It must not calculate a diagnostic reading.
