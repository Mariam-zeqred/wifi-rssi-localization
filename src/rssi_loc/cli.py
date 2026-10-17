"""`python -m rssi_loc download|run`. `run` writes results.json and the figures."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import data, evaluate, plots, synthetic
from .features import KINDS, transform
from .models import KnnLocalizer, MlpLocalizer

KS = [1, 2, 3, 5, 7, 10, 15]


def _load(args: argparse.Namespace) -> tuple[data.Fingerprints, data.Fingerprints, str]:
    if args.synthetic:
        train = synthetic.make_building(n=600, seed=0)
        test = synthetic.make_building(n=150, seed=1, noise_db=4.0)
        return train, test, "synthetic building (see synthetic.py); numbers are not meaningful"
    paths = data.download(Path(args.data), log=lambda s: print(s, file=sys.stderr))
    return (
        data.load(paths["trainingData.csv"]),
        data.load(paths["validationData.csv"]),
        ("UJIIndoorLoc: trainingData.csv -> validationData.csv (official split)"),
    )


def run(args: argparse.Namespace) -> dict:
    t0 = time.time()
    train, test, source = _load(args)
    x_tr, x_te = transform(train.rssi, args.features), transform(test.rssi, args.features)
    print(f"{source}: {len(train)} train, {len(test)} test, features={args.features}")
    out, figs = Path(args.out), Path(args.figures)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    res: dict = {
        "source": source,
        "n_train": len(train),
        "n_test": len(test),
        "features": args.features,
    }
    errs: dict[str, np.ndarray] = {}

    def make_knn(k: int = args.k) -> KnnLocalizer:
        return KnnLocalizer(k=k, metric=args.metric)

    def make_mlp() -> MlpLocalizer:
        return MlpLocalizer(hidden=args.hidden, seed=0)

    for name, make in (("knn", make_knn), ("mlp", make_mlp)):
        t = time.time()
        m_cv, e_cv = evaluate.cross_validate(make, x_tr, train, n_splits=args.folds)
        pred = evaluate.fit_predict(make(), x_tr, train, x_te)
        m_split = evaluate.summarise(pred, test)
        errs[f"{name} (CV)"], errs[name] = e_cv, evaluate.errors_m(pred.xy, test.xy)
        res[name] = {
            "cv": m_cv.as_dict(),
            "split": m_split.as_dict(),
            "seconds": round(time.time() - t, 1),
        }
        print(f"{name:4s} CV: {m_cv.as_dict()}\n{name:4s} split: {m_split.as_dict()}")
        plots.prediction_map(test.xy, pred.xy, figs / f"03_{name}_map.png", f"{name}: {source}")
    res["knn"]["k"], res["knn"]["metric"] = args.k, args.metric
    res["mlp"]["hidden"] = list(args.hidden)

    sweep = evaluate.k_sweep(KS, x_tr, train, x_te, test, make_knn)
    res["knn"]["k_sweep"] = sweep.to_dict(orient="records")
    print(sweep.to_string(index=False))

    # Which feature representation and which distance? kNN only, official split, same k.
    res["knn_variants"] = [
        {
            "features": kind,
            "metric": metric,
            **evaluate.summarise(
                evaluate.fit_predict(
                    KnnLocalizer(k=args.k, metric=metric),
                    transform(train.rssi, kind),
                    train,
                    transform(test.rssi, kind),
                ),
                test,
            ).as_dict(),
        }
        for kind in KINDS
        for metric in ("euclidean", "manhattan")
    ]
    print("knn variants:", *res["knn_variants"], sep="\n  ")

    stds = {
        "any phone": evaluate.rssi_repeatability(train),
        "same phone": evaluate.rssi_repeatability(train, True),
    }
    res["rssi_repeatability_db"] = {
        k: {
            "n_pairs": int(len(v)),
            "median_std": round(float(np.median(v)), 2),
            "p90_std": round(float(np.percentile(v, 90)), 2),
        }
        for k, v in stds.items()
    }
    print("rssi repeatability:", res["rssi_repeatability_db"])

    plots.error_cdf(errs, figs / "01_error_cdf.png")
    plots.k_sweep(sweep, figs / "02_k_sweep.png")
    plots.repeatability(stds, figs / "04_rssi_repeatability.png")
    res["seconds_total"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {out / 'results.json'} and {figs}/*.png in {res['seconds_total']} s")
    return res


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rssi_loc", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="fetch the two CSVs into --data and verify SHA-256")
    d.add_argument("--data", default="data")
    r = sub.add_parser("run", help="kNN + MLP, cross-validation and official split, figures")
    r.add_argument("--data", default="data")
    r.add_argument("--synthetic", action="store_true", help="use the synthetic building instead")
    r.add_argument("--out", default="results")
    r.add_argument("--figures", default="figures")
    r.add_argument("--features", default="powed", choices=KINDS)
    r.add_argument("--k", type=int, default=3)
    r.add_argument("--metric", default="euclidean", choices=["euclidean", "manhattan"])
    r.add_argument("--folds", type=int, default=5)
    r.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    args = p.parse_args(argv)
    if args.cmd == "download":
        data.download(Path(args.data))
    else:
        args.hidden = tuple(args.hidden)
        run(args)
