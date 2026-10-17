"""Error in metres, summary metrics, cross-validation and the RSSI repeatability measure."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .data import FLOOR_DBM, Fingerprints
from .models import Localizer, Prediction


@dataclass(frozen=True)
class Metrics:
    n: int
    mean_m: float
    median_m: float
    p75_m: float
    p95_m: float
    floor_acc: float
    building_acc: float

    def as_dict(self) -> dict[str, float]:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def errors_m(pred_xy: np.ndarray, true_xy: np.ndarray) -> np.ndarray:
    """2-D Euclidean error per fingerprint, in metres (UTM coordinates are metric)."""
    return np.linalg.norm(pred_xy - true_xy, axis=1)


def summarise(pred: Prediction, truth: Fingerprints) -> Metrics:
    e = errors_m(pred.xy, truth.xy)
    return Metrics(
        n=len(e),
        mean_m=float(e.mean()),
        median_m=float(np.median(e)),
        p75_m=float(np.percentile(e, 75)),
        p95_m=float(np.percentile(e, 95)),
        floor_acc=float((pred.floor == truth.floor).mean()),
        building_acc=float((pred.building == truth.building).mean()),
    )


def cdf(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted errors and the fraction of fingerprints with error <= each value."""
    e = np.sort(errors)
    return e, np.arange(1, len(e) + 1) / len(e)


def fit_predict(
    model: Localizer, x_train: np.ndarray, train: Fingerprints, x_test: np.ndarray
) -> Prediction:
    return model.fit(x_train, train.xy, train.floor, train.building).predict(x_test)


def cross_validate(
    make_model: Callable[[], Localizer],
    x: np.ndarray,
    fp: Fingerprints,
    n_splits: int = 5,
    seed: int = 0,
) -> tuple[Metrics, np.ndarray]:
    """K-fold over random fingerprints. Pooled out-of-fold predictions -> one Metrics.

    Note: reference points were sampled many times, so a random fold almost always contains
    fingerprints from the same spot as the held-out ones. This is the optimistic number; the
    official training/validation split (different users, phones and months) is the honest one.
    """
    pred_xy = np.zeros_like(fp.xy)
    pred_floor = np.zeros_like(fp.floor)
    pred_bld = np.zeros_like(fp.building)
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(x):
        sub = Fingerprints(fp.rssi[tr], fp.xy[tr], fp.floor[tr], fp.building[tr], fp.meta.iloc[tr])
        p = fit_predict(make_model(), x[tr], sub, x[te])
        pred_xy[te], pred_floor[te], pred_bld[te] = p.xy, p.floor, p.building
    pred = Prediction(pred_xy, pred_floor, pred_bld)
    return summarise(pred, fp), errors_m(pred.xy, fp.xy)


def rssi_repeatability(fp: Fingerprints, same_phone: bool = False, min_n: int = 5) -> np.ndarray:
    """Std of RSSI (dB) for every (reference point, access point) pair seen >= min_n times.

    Reference point = (building, floor, SPACEID, RELATIVEPOSITION), optionally also PHONEID.
    The spread measures how much the same access point's signal varies at the same spot: the
    noise floor any fingerprinting method has to live with.
    """
    keys = ["BUILDINGID", "FLOOR", "SPACEID", "RELATIVEPOSITION"] + (
        ["PHONEID"] if same_phone else []
    )
    df = fp.meta.copy()
    df["BUILDINGID"], df["FLOOR"] = fp.building, fp.floor
    group = df.groupby(keys).ngroup().to_numpy()
    seen = fp.rssi > FLOOR_DBM
    stds = []
    for g in np.unique(group):
        rows = fp.rssi[group == g]
        count = seen[group == g].sum(axis=0)
        for ap in np.flatnonzero(count >= min_n):
            v = rows[:, ap]
            stds.append(v[v > FLOOR_DBM].std())
    return np.asarray(stds)


def k_sweep(
    ks: list[int],
    x_train: np.ndarray,
    train: Fingerprints,
    x_test: np.ndarray,
    test: Fingerprints,
    make_model: Callable[[int], Localizer],
) -> pd.DataFrame:
    rows = [
        {"k": k, **summarise(fit_predict(make_model(k), x_train, train, x_test), test).as_dict()}
        for k in ks
    ]
    return pd.DataFrame(rows)
