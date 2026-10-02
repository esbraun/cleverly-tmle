# Gate S5: house rules and shared design (commit a8661971)

Verdict: PASS WITH REQUIRED FIXES. Four wording fixes below. No structural or scope problem.

Checks run: `git show a8661971`; `pytest -q -p no:cacheprovider tests/unit/test_documentation_links.py -k "example or index"` (67 passed); `python -m tests.prose --path` on both pages (exit 0, one pre-existing 30-word advisory at line 331, not from this commit); API names and defaults read in `src/cleverly/sensitivity/{omitted_variable,evalue,missingness,simulated_confounding}.py` and `src/cleverly/assessment.py`; generators read in `src/cleverly/datasets/synthetic.py` and `longitudinal.py`; executed probes in `reviews/notebook-review/refusal-inventory.md`.

## OK

| item | evidence |
| --- | --- |
| no refusal instruction remains; outline rows 11 and 12 match the new sections and their anchors | page read end to end; the only "not implemented" mention is the two-claims reason, which is about wording, not a refusal step |
| `robustness_value()`, `benchmark()`, `omitted_confounding()`, `evalue()`, `simulated_confounding()`, `missingness()`, `tipping_gamma()`, `ConfounderStrengthGrid(treatment=..., outcome=...)` exist under those names on `result.sensitivity` and in `cleverly.sensitivity`; default `nu2_estimator` resolves to `doubly_robust` | `assessment.py:4035-4231`, `omitted_variable.py:518,571,966` |
| `tipping_gamma` default `use_ci=False` searches the point estimate, so "the tilt at which the estimate reaches its null" is right | `missingness.py:491-530` |
| `evalue()` runs on an unweighted DR-TMLE fit with a Gaussian ATE and marks the result approximate; reads estimate and interval only | `validation-methods.md:1533-1538`, `evalue.py:628-645`, probe `p_dr.py` (RR 1.888, E 3.18) |
| `simulated_confounding()` supports a binary-treatment collaborative ATE and regime, shift, incremental targets; outcome-axis cells fail on a `q_bounds` fit so a treatment-only grid is needed | `validation-methods.md:1104,1110-1114,1162`; inventory rows 109 and 145 (0 failures with `outcome=(0.0,)`) |
| omitted-variable bound refused on DR-TMLE, C-TMLE, response-mechanism, longitudinal, and `msm`/`regime`/`shift` axes, so the table routes each kind of fit away from it correctly | `validation-methods.md:760-777` |
| HGB rule: sklearn 1.9 `early_stopping="auto"` enables early stopping only for n > 10,000; a bare HGB propensity gave a negative doubly robust nu^2 | sklearn default checked in the venv; `learner-experiment.md:66` |
| covariates: every generator the notebooks use emits i.i.d. standard-normal baseline covariates untransformed (`_latent` at `synthetic.py:367`, `payload[name] = latent[:, index]`; `w1, w2 = rng.standard_normal(n)` in all three longitudinal laws). The clustered law's shared latent is not emitted | source read |
| "death in the data" row and composite coding agree with the longitudinal protocol (death before day 30 is not top box) and with the survival notebook, which codes death as an event in the main fit | `index.md` rows; `longitudinal-survival.ipynb` `failure-mode` cell |
| sweep rules (60 draws, 90% "on this draw", `UNPRINTED_DECIMALS` source) implement R3, R4, R8 as the plan words them | `plan.md:15-20` |
| trust checklist implements R5 (law, nuisance construction, outcome type, fold layer, cell role) | `plan.md:17` |
| style: no em dash or `--` in added prose; new content is tabular; sentences are single-idea | diff grep |

## REQUIRED FIXES

1. `example-notebooks.md`, sensitivity table, MSM row. The saturated fit is an `MSMProjection` fit, and its coefficients sit on the `msm` axis, which `omitted_variable.py` refuses. The notebook's bound runs on a separate multi-arm `ATE(reference="low")` fit (`msm-projections.ipynb`, cells `saturated` and `sensitivity`). Replace the row with:

   `| MSM projection | robustness_value(estimand="ate[medium vs low]") and its sibling on the multi-arm ATE(reference="low") fit, whose contrasts share the influence curves of the saturated MSM coefficients | a bound on each arm contrast, not on the MSM coefficient. The msm axis itself is refused |`

2. `example-notebooks.md`, sensitivity table, DR-TMLE row. The printed note is a three-step chain, not one conversion: "Standardised by sd(Y)", Chinn's log(OR)/1.81 to an odds ratio, then the common-outcome square root to a risk ratio (`evalue.py:638-645`). Replace "The output prints its conversion from the standardized difference to a risk ratio" with: "The output prints its chain: the standardized mean difference, Chinn's step to an odds ratio, and the common-outcome square root to a risk ratio."

3. `example-notebooks.md`, sensitivity section intro. "derives each one" overclaims: the technical reference states that no source defines the simulated-confounding displacement (`validation-methods.md:942`), and the longitudinal drop-one refit is a benchmark the page itself defines. Replace the sentence with: "The sensitivity section of the technical reference describes each one and says which of them is a bound."

   Same table, first row, last cell: replace "it is finite only when the inverse propensity has a finite mean" with "it is finite only when 1/g and 1/(1 - g) both have finite means" (for the ATE, nu^2 = E[1/g + 1/(1 - g)]).

4. `index.md`, shared design table and program table.
   - "missing at random" row. "holds only when no patient dies" is not the condition: it also holds when every survivor responds, and the real point is that response is always 1 for a death, so it depends on the composite outcome. Replace with: `| missing at random | a death is always observed, so response depends on the composite outcome. Missing at random given treatment and baseline covariates fails when some patients die and some survivors do not respond. The analysis needs missing at random among survivors, given treatment, covariates, and survival |`
   - Time-to-event failure-mode cell. A risk is not an effect, and identification is separate from the estimand. Replace with: "coding death as censoring estimates the readmission risk under a hypothetical removal of death. Its arm contrast is a controlled direct effect. The censoring fit identifies it only under exchangeability and positivity for death given the recorded history, and it is not the reported total effect, which leaves death alone".

## Optional

- `index.md`: "Real covariates would be correlated." has no evidence behind it; delete it or drop "would be" for "are usually". The new "death in the data" row repeats the "unreturned surveys" row; merge or keep deliberately.
- `index.md`: re-wrap the long covariate paragraph line.

## Re-check of 84d26a6b

Verdict: PASS.

Checks run: `git show 84d26a6b` (two files, 6 lines changed, no other edits); `python -m tests.prose --path` on both pages (exit 0, only the pre-existing line 331 advisory); `msm-projections.ipynb` grep confirms `ATE(reference="low")` and `robustness_value(estimand=name)` over `ate[medium vs low]` and `ate[high vs low]`; `omitted_variable.py:161` lists `msm` among refused axes.

| fix | applied as specified | new error |
| --- | --- | --- |
| 1 MSM row | yes, verbatim | none. Saturated MSM coefficients under reference coding equal the arm contrasts, so the shared influence curve claim holds |
| 2 DR-TMLE chain | yes, verbatim | none |
| 3 intro and nu^2 condition | yes, verbatim; `$1/g$` and `$1/(1 - g)$` in math | none |
| 4 index.md rows | yes, verbatim | none. No em dash or `--`; active voice |

Optional items from the first gate remain open and are not required.
