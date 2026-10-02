import numpy as np


def true_terms(x):
    """W1, W2^2, W2*W3, 1(W4>0) from columns ordered (W1, W2, W3, W4)."""
    x = np.asarray(x, dtype=float)
    return np.column_stack([x[:, 0], x[:, 1] ** 2, x[:, 1] * x[:, 2], (x[:, 3] > 0).astype(float)])
