"""The four figures. Each function takes arrays and a path; nothing here computes a result."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .evaluate import cdf  # noqa: E402


def error_cdf(curves: dict[str, np.ndarray], path: Path, x_max: float | None = None) -> None:
    """One CDF line per named error array; dashed names (ending in '(CV)') are cross-validation."""
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for name, err in curves.items():
        e, p = cdf(err)
        ax.plot(
            e, p, ls="--" if "CV" in name else "-", label=f"{name}, median {np.median(e):.1f} m"
        )
    ax.set_xlim(0, x_max or max(np.percentile(e, 98) for e in curves.values()))
    ax.set_ylim(0, 1)
    ax.set_xlabel("positioning error (m)")
    ax.set_ylabel("fraction of fingerprints with error ≤ x")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def k_sweep(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(table["k"], table["mean_m"], "o-", label="mean error")
    ax.plot(table["k"], table["median_m"], "s-", label="median error")
    ax.plot(table["k"], table["p75_m"], "^-", label="75th percentile")
    ax.set_xlabel("k (neighbours)")
    ax.set_ylabel("error (m)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def prediction_map(true_xy: np.ndarray, pred_xy: np.ndarray, path: Path, title: str) -> None:
    """True position (dot) joined to its prediction (line), coloured by error."""
    err = np.linalg.norm(pred_xy - true_xy, axis=1)
    fig, ax = plt.subplots(figsize=(7, 5))
    order = np.argsort(err)
    cmap = plt.get_cmap("viridis")
    vmax = np.percentile(err, 95)
    for i in order:
        c = cmap(min(err[i] / vmax, 1.0))
        ax.plot([true_xy[i, 0], pred_xy[i, 0]], [true_xy[i, 1], pred_xy[i, 1]], color=c, lw=0.6)
    sc = ax.scatter(true_xy[:, 0], true_xy[:, 1], c=err, s=6, cmap=cmap, vmin=0, vmax=vmax)
    fig.colorbar(sc, ax=ax, label="error (m)")
    ax.set_aspect("equal")
    ax.set_xlabel("easting (m, UTM)")
    ax.set_ylabel("northing (m, UTM)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def repeatability(stds: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bins = np.arange(0, 16, 0.5)
    for name, s in stds.items():
        ax.hist(
            np.clip(s, 0, 15.9), bins=bins, alpha=0.55, density=True, label=f"{name} (n = {len(s)})"
        )
    ax.set_xlabel("std of RSSI at one spot from one access point (dB)")
    ax.set_ylabel("density")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
