"""A small synthetic building for tests and the smoke run (no download needed).

Access points are placed at random on each floor; RSSI follows the log-distance path-loss
model with Gaussian noise, plus a per-floor penalty. Values below the detection threshold are
coded as NOT_DETECTED, exactly like the real CSVs. This is a stand-in for the code path only:
its errors say nothing about real buildings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import META_COLS, N_WAP, NOT_DETECTED, Fingerprints, from_frame


def make_building(
    n: int = 600,
    n_ap: int = 30,
    size_m: tuple[float, float] = (80.0, 40.0),
    floors: int = 2,
    noise_db: float = 3.0,
    seed: int = 0,
) -> Fingerprints:
    rng = np.random.default_rng(seed)
    ap_xy = rng.uniform((0, 0), size_m, size=(n_ap, 2))
    ap_floor = rng.integers(0, floors, size=n_ap)
    xy = rng.uniform((0, 0), size_m, size=(n, 2))
    floor = rng.integers(0, floors, size=n)
    d = np.hypot(*(xy[:, None, :] - ap_xy[None, :, :]).transpose(2, 0, 1))  # (n, n_ap)
    rssi = -40 - 10 * 2.5 * np.log10(np.maximum(d, 1.0))  # log-distance, n = 2.5
    rssi -= 15 * np.abs(floor[:, None] - ap_floor[None, :])  # floor penalty
    rssi += rng.normal(0, noise_db, size=rssi.shape)
    rssi = np.where(rssi < -100, NOT_DETECTED, np.round(rssi))
    waps = np.full((n, N_WAP), NOT_DETECTED, dtype=float)
    waps[:, :n_ap] = rssi
    df = pd.DataFrame(waps, columns=[f"WAP{i:03d}" for i in range(1, N_WAP + 1)])
    df["LONGITUDE"], df["LATITUDE"] = xy[:, 0], xy[:, 1]
    df["FLOOR"], df["BUILDINGID"] = floor, 0
    df["SPACEID"] = (xy[:, 0] // 10 * 100 + xy[:, 1] // 10).astype(int)
    df["RELATIVEPOSITION"], df["USERID"] = 1, 1
    df["PHONEID"], df["TIMESTAMP"] = rng.integers(0, 3, size=n), np.arange(n)
    return from_frame(df[[c for c in df.columns if c not in META_COLS] + META_COLS])
