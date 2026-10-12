"""Aligning two implementations on the replications they actually share.

Pivoting each measured column on its own and dropping missing rows column by column lets the
resulting arrays disagree about *which* replications they hold, so a paired statistic ends up
comparing implementation A's replication 7 with implementation B's replication 8 -- or, when
the two survivors' counts differ, indexing past the end of the shorter one.  One pivot, one
drop, one count.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Paired:
    """Columns of two implementations, aligned on a common set of replications."""

    frame: pd.DataFrame
    dropped: int

    def __len__(self) -> int:
        return len(self.frame)

    def column(self, name: str, implementation: str) -> np.ndarray:
        return self.frame[(name, implementation)].to_numpy(dtype=float)

    def arrays(
        self, columns: Sequence[str], implementations: Sequence[str]
    ) -> dict[str, np.ndarray]:
        """A flat ``{"estimate_cleverly": ...}`` mapping, for the bootstrap engine."""
        return {
            f"{column}_{implementation}": self.column(column, implementation)
            for column in columns
            for implementation in implementations
        }


def paired_wide(
    group: pd.DataFrame,
    columns: Sequence[str],
    *,
    implementations: Sequence[str],
) -> Paired:
    """Pivot ``columns`` for every implementation at once and align on complete replications.

    A replication one implementation lacks is dropped from the pairing and counted in
    ``dropped``.  Only a recorded failure can leave one out: ``validate_replicates`` refuses any
    other missing row and every non-finite value before a pairing is formed (RM39 part a2).
    """
    wide = group.pivot(index="replicate", columns="implementation", values=list(columns))
    missing = [
        (column, implementation)
        for column in columns
        for implementation in implementations
        if (column, implementation) not in wide.columns
    ]
    if missing:
        raise KeyError(f"paired columns absent from the replication rows: {missing}")
    wide = wide.loc[:, [(column, name) for column in columns for name in implementations]]
    complete = wide.replace([np.inf, -np.inf], np.nan).dropna()
    dropped = len(wide) - len(complete)
    if len(complete) < 2:
        raise ValueError("fewer than two paired replications survive")
    return Paired(frame=complete, dropped=dropped)
