# Time until an event

These tutorials estimate the risk of an event, such as readmission, by a given day, when some people
leave follow-up and another event, such as death, can come first.

| your situation | example | the mistake it shows |
| --- | --- | --- |
| time until an event, when another event such as death can come first and some people leave follow-up | [readmission when death can come first](../longitudinal-survival.ipynb) | treating death like dropout estimates readmission in a world where nobody dies, which the program cannot bring about |
| one treatment at discharge and an event such as readmission by day 30, 60 and 90, when some patients leave follow-up, or patients who share a team over two decisions | [readmission by day 30, 60 and 90](../discharge-to-readmission.ipynb) | treating patients of one team as independent ignores that their risks move together, so the interval is too narrow |
| a treatment with two parts that act on two competing events, such as a medicine that lowers readmission through one part and raises the risk of death through another | [a medicine with two effects](../separable-effects.ipynb) | the drop in readmission under the whole medicine includes patients who died before they could be readmitted, so reading it as the benefit of the part that acts on readmission overstates that benefit |

```{toctree}
:hidden:
:maxdepth: 1

/examples/longitudinal-survival
/examples/discharge-to-readmission
/examples/separable-effects
```
