"""Odds <-> probability helpers."""
from __future__ import annotations

import numpy as np


def implied_prob(decimal_odds: np.ndarray | float) -> np.ndarray:
    o = np.asarray(decimal_odds, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(o > 1.0, 1.0 / o, np.nan)


def remove_vig_proportional(probs: np.ndarray) -> np.ndarray:
    """Normalise a set of implied probs so they sum to 1 (proportional / multiplicative method).

    `probs` shape (..., k) over the k outcomes of one market at one book.
    """
    p = np.asarray(probs, dtype=float)
    s = np.nansum(p, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = p / s
    return out


def overround(probs: np.ndarray) -> np.ndarray:
    """Book margin: sum of implied probs minus 1 (a.k.a. vig / juice)."""
    return np.nansum(np.asarray(probs, dtype=float), axis=-1) - 1.0
