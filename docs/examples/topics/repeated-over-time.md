# Repeated actions over time

These tutorials follow a treatment that is decided again and again, such as a weekly offer or a
monthly price, where earlier treatments change later behavior and later decisions.

| your situation | example | the mistake it shows |
| --- | --- | --- |
| a treatment repeated weekly, where earlier offers change later behavior | [weekly offer policies](../sequential-offers.ipynb) | treating two policies as if they had been tried on separate customers, when both are estimated from the same customers, overstates the uncertainty of their difference |
| an outcome recorded every week, such as whether a customer ordered, while the treatment repeats | [weekly orders under repeated offers](../weekly-outcomes.ipynb) | fitting each week without the earlier orders leaves out what drove later offers, and adding the weekly variances as if the weeks were independent makes the interval for the total slightly too narrow |
| two decisions in a row, such as an offer at discharge and another at day 14, where you want a rule for each decision learned from the records | [two decisions after discharge](../two-decision-rules.ipynb) | patients who got both offers stayed home more often than both offers for everyone would achieve, because who got each offer depended on readiness and on a score the first offer changed. The best first rule also depends on the second: judged alone, the first offer looks helpful for patients the best pair of rules leaves without it |
| repeated treatments where you can model how the whole process unfolds | [three offers after discharge](../model-based-over-time.ipynb) | a model of the process that is too simple gives a confident wrong answer; comparing two methods that rest on different models shows the problem |
| a price or dose that varies continuously and is set again each month | [monthly prices](../sequential-pricing.ipynb) | one regression that holds later behavior and later prices fixed misses part of the effect of a price rise |

## See also

- [Readmission by day 30, 60 and 90: longitudinal TMLE](../discharge-to-readmission.ipynb) includes a program with two decisions over time, made by teams whose patients share part of their risk.

```{toctree}
:hidden:
:maxdepth: 1

Weekly offer policies </examples/sequential-offers>
Weekly orders under repeated offers </examples/weekly-outcomes>
Two decisions after discharge </examples/two-decision-rules>
Three offers after discharge </examples/model-based-over-time>
Monthly prices </examples/sequential-pricing>
```
