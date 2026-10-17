"""Download (with SHA-256 check) and load the two UJIIndoorLoc CSV files.

Dataset: Torres-Sospedra et al., "UJIIndoorLoc: A new multi-building and multi-floor database
for WLAN fingerprint-based indoor localization problems", IPIN 2014. UCI ML Repository id 310
(CC BY 4.0). Each row is one fingerprint: 520 RSSI values in dBm (100 = access point not seen),
the UTM position in metres, floor, building and who/what/when took it.

The expected hashes were taken from three independent public copies of the files that are
byte-identical to each other and have the row counts UCI documents (19 937 and 1 111). The UCI
archive is tried first; the copies are fallbacks.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NOT_DETECTED = 100  # the dataset's code for "access point not seen"
FLOOR_DBM = -105  # replaces NOT_DETECTED: 1 dB below the weakest value in the data (-104)
N_WAP = 520
WAP_COLS = [f"WAP{i:03d}" for i in range(1, N_WAP + 1)]
META_COLS = ["SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP"]

SHA256 = {
    "trainingData.csv": "45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168",
    "validationData.csv": "5f90c536648cd657b2c516d20c4e0968d4003279ea6bd5d5d5322d3f1e8905c0",
}
UCI_ZIP = "https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip"
_RAW = "https://raw.githubusercontent.com/"
# Each source is either {"zip": url} (both CSVs inside) or {file name: url} for bare CSVs.
SOURCES: list[dict[str, str]] = [
    {"zip": UCI_ZIP},
    {
        "trainingData.csv": _RAW
        + "litanli/indoor-locationing/master/UJIIndoorLoc_trainingData.csv",
        "validationData.csv": _RAW
        + "litanli/indoor-locationing/master/UJIIndoorLoc_validationData.csv",
    },
    {
        "trainingData.csv": _RAW + "vk1408/UJIIndoorLoc/main/csv/trainingData.csv",
        "validationData.csv": _RAW + "vk1408/UJIIndoorLoc/main/csv/validationData.csv",
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verified(path: Path, expected: str) -> bool:
    return path.is_file() and sha256(path) == expected


def _get(url: str, timeout: float = 300) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (fixed https URLs)
        return r.read()


def download(
    data_dir: Path,
    sources: list[dict[str, str]] = SOURCES,
    expected: dict[str, str] = SHA256,
    log: Callable[[str], None] = print,
) -> dict[str, Path]:
    """Make sure every file in `expected` exists in `data_dir` with the right hash.

    Tries each source in order and stops as soon as all files verify. A file whose hash does
    not match is reported and not used. Raises RuntimeError if any file is still missing.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: data_dir / name for name in expected}
    missing = {n for n, p in paths.items() if not verified(p, expected[n])}
    for n in expected.keys() - missing:
        log(f"{n}: present, SHA-256 ok")
    for source in sources:
        if not missing:
            break
        try:
            got: dict[str, bytes] = {}
            if "zip" in source:
                log(f"fetching {source['zip']}")
                with zipfile.ZipFile(io.BytesIO(_get(source["zip"]))) as z:
                    for member in z.namelist():
                        name = member.rsplit("/", 1)[-1]
                        if name in missing:
                            got[name] = z.read(member)
            else:
                for name in sorted(missing & source.keys()):
                    log(f"fetching {source[name]}")
                    got[name] = _get(source[name])
        except Exception as e:  # noqa: BLE001 (a failed source is not fatal)
            log(f"  failed: {e}")
            continue
        for name, blob in got.items():
            digest = hashlib.sha256(blob).hexdigest()
            if digest == expected[name]:
                paths[name].write_bytes(blob)
                missing.discard(name)
                log(f"  {name}: {len(blob) / 1e6:.1f} MB, SHA-256 ok")
            else:
                log(f"  {name}: SHA-256 mismatch ({digest[:12]}...), discarded")
    if missing:
        raise RuntimeError(f"could not obtain {sorted(missing)}; see the log above")
    return paths


@dataclass(frozen=True)
class Fingerprints:
    """One fingerprint per row. `rssi` is in dBm with NOT_DETECTED replaced by FLOOR_DBM."""

    rssi: np.ndarray  # (n, 520) float
    xy: np.ndarray  # (n, 2) UTM metres: longitude (easting), latitude (northing)
    floor: np.ndarray  # (n,) int 0..4
    building: np.ndarray  # (n,) int 0..2
    meta: pd.DataFrame  # META_COLS, n rows

    def __len__(self) -> int:
        return len(self.floor)


def from_frame(df: pd.DataFrame) -> Fingerprints:
    rssi = np.array(df[WAP_COLS].to_numpy(dtype=float))  # copy: pandas 3 may return a view
    rssi[rssi == NOT_DETECTED] = FLOOR_DBM
    meta = df[[c for c in META_COLS if c in df.columns]].reset_index(drop=True)
    return Fingerprints(
        rssi=rssi,
        xy=df[["LONGITUDE", "LATITUDE"]].to_numpy(dtype=float),
        floor=df["FLOOR"].to_numpy(dtype=int),
        building=df["BUILDINGID"].to_numpy(dtype=int),
        meta=meta,
    )


def load(path: Path) -> Fingerprints:
    return from_frame(pd.read_csv(path))
