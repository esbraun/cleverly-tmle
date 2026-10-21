# Repeated point-treatment cross-fitted TMLE

This directory freezes an independent repeated-sampling study of `cleverly`'s repeated,
stacked point-treatment CV-TMLE. The primary study uses five unstratified folds per
draw, three fold draws, pooled targeting, the median of the draw-specific point estimates,
and the median of within-draw variance plus squared split displacement. Risk ratios and odds
ratios are aggregated on their log inference scale.

Each draw's split reads neither the treatment nor the outcome, and the fit reads the realized
plan back rather than trusting the keyword. The continuous law draws a proportion, so the study
declares `q_bounds=(0, 1)` for it. The binary law is passed no `q_bounds`, because a binary
outcome already sits on the unit interval.

The study tests arm means, ATE, ATT, ATC, observed mean, PAR, and, where defined, PAF, RR, and OR
against exact truth under binary and bounded continuous outcome laws. It reuses the shared
CV-TMLE property machinery for double robustness, empirical and reported root-n rates, interval
calibration, type-I error, and power. A separate property family holds one binary-law sample fixed
across 400 labelled fold-seed trials. Each trial fits the three-draw median and draws 0 and 1
alone. The positive rows compare the ATE spread from three draws with the spread of draw 0
(`three_repeats` and its base arm `one_repeat`). The control `second_draw_control` compares the
spread of draw 1 with the spread of draw 0, a ratio whose true value is 1, and passes when the
upper endpoint of its 99% interval reaches 1.

No full estimator is compared. Chernozhukov et al. establish the median aggregation rule and
fixed-repeat first-order validity, and zEpid implements the same aggregation formula. zEpid's
nuisance training and foldwise targeting differ from `cleverly`'s complement-trained stacked
pooled update, so it is corroboration for the reporting layer rather than numerical parity for
the complete estimator. `equivalence.csv` is intentionally empty and schema-valid.

Regenerate from the repository root:

```powershell
uv run --extra dev python -m tests.canonical.repeated_crossfit.regenerate
```

## Rerun for the second-draw control

The study has no inherit path. The rerun for the `repeat_stability` control is the whole declared
run. Run it from a clean, pushed commit with every thread variable set to 1:

```bash
python -m tests.canonical.repeated_crossfit.regenerate --jobs 4 --output ~/scratch/x79-point
```

Only the `repeat_stability` rows may change beyond float noise. Compare every other row of the new
`properties.csv` with the committed one before the commit. Discrete columns and verdicts must
match. A float column must agree within 1e-9, and the largest difference goes in the pull request.
`tests/canonical/provenance-revisions.md` declares this reading.

The declarations are in `tests/studies/repeated_crossfit.py` and
`tests/studies/repeated_crossfit_properties.py`. The reader-facing scope, measurements, and
limitations are in
[`docs/technical-reference/method-evidence/repeated-cross-fitting.md`](../../../docs/technical-reference/method-evidence/repeated-cross-fitting.md).
The manifest records all margins, seeds, configuration, source hashes, and artifact hashes.
