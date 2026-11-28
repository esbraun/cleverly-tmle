# Risk of an event by a given day (churn, readmission)

These tutorials estimate the risk of an event by a given day, when some people leave follow-up and
another event, such as death, can come first. They follow hospital readmission; the same analysis
applies to the time until a customer churns.

| your situation | example | the mistake it shows |
| --- | --- | --- |
| time until an event, when another event such as death can come first and some people leave follow-up | [readmission when death can come first](../longitudinal-survival.ipynb) | treating death like dropout estimates readmission in a world where nobody dies, which the program cannot bring about |
| one treatment at discharge and an event such as readmission by day 30, 60 and 90, when some patients leave follow-up, or patients who share a team over two decisions | [readmission by day 30, 60 and 90](../discharge-to-readmission.ipynb) | treating patients of one team as independent ignores that their risks move together, so the interval is too narrow |
| a treatment with two parts that act on two competing events, such as a medicine that lowers readmission through one part and raises the risk of death through another | [a medicine with two effects](../separable-effects.ipynb) | the drop in readmission under the whole medicine includes patients who died before they could be readmitted, so reading it as the benefit of the part that acts on readmission overstates that benefit |
| one treatment at discharge, and the exact time of readmission, of death first or of leaving follow-up, with no check days | [readmission or death with exact times](../exact-event-times.ipynb) | navigation went to patients more likely to be readmitted, so the share readmitted among offered patients understates how much the offer lowers readmission |

```{toctree}
:hidden:
:maxdepth: 1

Readmission when death can come first </examples/longitudinal-survival>
Readmission by day 30, 60 and 90 </examples/discharge-to-readmission>
A medicine with two effects on readmission and death </examples/separable-effects>
Readmission or death after discharge, with exact event times </examples/exact-event-times>
```
