"""Two localisers with one interface: fit on fingerprints, predict position, floor, building."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier, MLPRegressor

N_FLOORS = 5  # floors per building in UJIIndoorLoc (0..4), used to pack (building, floor)


@dataclass(frozen=True)
class Prediction:
    xy: np.ndarray  # (n, 2) metres
    floor: np.ndarray  # (n,) int
    building: np.ndarray  # (n,) int


class Localizer(Protocol):
    def fit(self, x: np.ndarray, xy: np.ndarray, floor: np.ndarray, building: np.ndarray): ...
    def predict(self, x: np.ndarray) -> Prediction: ...


class KnnLocalizer:
    """Weighted k-nearest-neighbour fingerprinting.

    Position = inverse-distance weighted mean of the k nearest reference fingerprints;
    floor and building = weighted majority vote over the same k neighbours.
    Manhattan distance is the classic choice (Torres-Sospedra 2015); Euclidean did better on
    the official split here (see results.json "knn_variants"), so it is the default.
    """

    def __init__(self, k: int = 3, metric: str = "euclidean", weighted: bool = True):
        self.k, self.metric, self.weighted = k, metric, weighted
        self._nn = NearestNeighbors(n_neighbors=k, metric=metric)

    def fit(self, x: np.ndarray, xy: np.ndarray, floor: np.ndarray, building: np.ndarray):
        self._nn.fit(x)
        self._xy, self._floor, self._building = xy, floor, building
        return self

    def predict(self, x: np.ndarray) -> Prediction:
        dist, idx = self._nn.kneighbors(x)  # (n, k)
        w = 1.0 / (dist + 1e-6) if self.weighted else np.ones_like(dist)
        w /= w.sum(axis=1, keepdims=True)
        xy = np.einsum("nk,nkd->nd", w, self._xy[idx])
        label = self._building[idx] * N_FLOORS + self._floor[idx]
        votes = np.zeros((len(x), 3 * N_FLOORS))
        np.add.at(votes, (np.arange(len(x))[:, None], label), w)
        best = votes.argmax(axis=1)
        return Prediction(xy=xy, floor=best % N_FLOORS, building=best // N_FLOORS)


class MlpLocalizer:
    """A small multilayer perceptron: one regressor for (x, y), one classifier for the floor.

    Targets are standardised so the loss treats easting and northing alike. Early stopping on
    a 10 % hold-out of the training fingerprints keeps the run short and limits overfitting.
    """

    def __init__(self, hidden: tuple[int, ...] = (256, 128), seed: int = 0, max_iter: int = 300):
        common = dict(
            hidden_layer_sizes=hidden,
            alpha=1e-4,
            early_stopping=True,
            max_iter=max_iter,
            random_state=seed,
        )
        self._reg = MLPRegressor(**common)
        self._clf = MLPClassifier(**common)

    def fit(self, x: np.ndarray, xy: np.ndarray, floor: np.ndarray, building: np.ndarray):
        self._mu, self._sd = xy.mean(axis=0), xy.std(axis=0)
        self._reg.fit(x, (xy - self._mu) / self._sd)
        self._clf.fit(x, building * N_FLOORS + floor)
        return self

    def predict(self, x: np.ndarray) -> Prediction:
        xy = self._reg.predict(x) * self._sd + self._mu
        label = self._clf.predict(x)
        return Prediction(xy=xy, floor=label % N_FLOORS, building=label // N_FLOORS)
