# Learn whom to treat from the data

This tutorial learns a rule that decides who gets the treatment, from recorded characteristics, and
estimates how well the rule would do on new people. It also shows a random offer when there are
fewer places than people who qualify.

| your situation | example | the mistake it shows |
| --- | --- | --- |
| a treatment that helps some patients and harms others, so you must decide who gets it, or share a limited number of places at random | [who should get the offer](../treatment-rules.ipynb) | a rule scored on the same patients it was learned from looks better than it will do on new patients |

## See also

- [Two decisions after discharge: a learned treatment rule with longitudinal TMLE](../two-decision-rules.ipynb) learns a rule for each of two decisions in a row.
- [Who gains most from a coupon: an effect that changes with age](../conditional-effects.ipynb) also learns whom to send a coupon from how the effect changes with age.

```{toctree}
:hidden:
:maxdepth: 1

Who should get the offer </examples/treatment-rules>
```
