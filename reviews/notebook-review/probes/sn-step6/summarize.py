"""Summarize probe-s1000.csv (the sweep's seeds) and probe-s2000.csv (disjoint seeds).

Usage: ``python summarize.py``; writes ``summary.log``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRUE_ATE = 1.2


def describe(frame, label):
    lines = []
    m = len(frame)
    for est in ("lib", "oracle"):
        c = frame[f"{est}_covers"]
        p = c.mean()
        h = 1.96 * np.sqrt(p * (1 - p) / m)
        sd = frame[f"{est}_psi"].std()
        se = frame[f"{est}_se"].mean()
        bias = frame[f"{est}_psi"].mean() - TRUE_ATE
        lines.append(
            f"{label} {est}: covers {c.sum()} of {m} ({p:.3f}; Wald 95% {p - h:.3f} to {p + h:.3f}); "
            f"bias {bias:+.4f} (MC SE {sd / np.sqrt(m):.4f}); emp SD {sd:.4f}; mean SE {se:.4f}; "
            f"SE/SD {se / sd:.3f}"
        )
    lines.append(
        f"{label} corr(lib_psi, oracle_psi) {frame['lib_psi'].corr(frame['oracle_psi']):.3f}"
    )
    blocks = [int(frame["lib_covers"].iloc[i : i + 500].sum()) for i in range(0, m, 500)]
    if len(blocks) > 1:
        lines.append(f"{label} lib covers per 500-seed block: {blocks}")
    return lines


s1000 = pd.read_csv(HERE / "probe-s1000.csv")
s2000 = pd.read_csv(HERE / "probe-s2000.csv")
lines = describe(s1000, "seeds 1000-1499")
lines += describe(s2000, "seeds 2000-4999")
lines += describe(pd.concat([s1000, s2000]), "pooled 3500")
text = "\n".join(lines) + "\n"
(HERE / "summary.log").write_text(text, encoding="utf-8", newline="\n")
print(text)
