"""RSSI → feature vector. All transforms map dBm to [0, 1] with "not seen" at 0."""

from __future__ import annotations

import numpy as np

from .data import FLOOR_DBM

KINDS = ("positive", "powed")


def positive(rssi: np.ndarray, floor_dbm: float = FLOOR_DBM) -> np.ndarray:
    """Shift so that `floor_dbm` (not detected) becomes 0 and 0 dBm becomes 1."""
    return np.clip((rssi - floor_dbm) / -floor_dbm, 0.0, 1.0)


def powed(rssi: np.ndarray, exponent: float = np.e, floor_dbm: float = FLOOR_DBM) -> np.ndarray:
    """`positive` raised to a power: strong signals count more, weak ones nearly vanish.

    Torres-Sospedra et al. (2015, Expert Systems with Applications) found this "powed"
    representation beats the plain one for kNN; the exponent e is their choice.
    """
    return positive(rssi, floor_dbm) ** exponent


def transform(rssi: np.ndarray, kind: str = "powed") -> np.ndarray:
    if kind == "positive":
        return positive(rssi)
    if kind == "powed":
        return powed(rssi)
    raise ValueError(f"unknown feature kind {kind!r}; choose from {KINDS}")
