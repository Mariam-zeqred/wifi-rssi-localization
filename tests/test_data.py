import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from rssi_loc import data
from rssi_loc.features import positive, powed, transform


def test_from_frame_replaces_not_detected(building):
    assert building.rssi.shape[1] == data.N_WAP
    assert not np.any(building.rssi == data.NOT_DETECTED)
    assert building.rssi.min() == data.FLOOR_DBM
    assert building.xy.shape == (len(building), 2)
    assert list(building.meta.columns) == data.META_COLS


def test_features_in_unit_interval(building):
    for kind in ("positive", "powed"):
        x = transform(building.rssi, kind)
        assert x.min() >= 0 and x.max() <= 1
        assert np.all(x[building.rssi == data.FLOOR_DBM] == 0)
    assert positive(np.array([[0.0]]))[0, 0] == 1.0
    assert powed(np.array([[-52.5]]))[0, 0] == pytest.approx(0.5**np.e)
    with pytest.raises(ValueError):
        transform(building.rssi, "nope")


def _fake_sources(blobs: dict[str, bytes]):
    """Serve `blobs` from an in-memory zip (source 0) and as bare files (source 1)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, b in blobs.items():
            z.writestr(f"UJIndoorLoc/{name}", b)
    served = {"zip://x": buf.getvalue(), **{f"raw://{n}": b for n, b in blobs.items()}}
    sources = [{"zip": "zip://x"}, {n: f"raw://{n}" for n in blobs}]
    return served, sources


def test_download_verifies_hash_and_falls_back(tmp_path, monkeypatch):
    good = {"a.csv": b"alpha\n", "b.csv": b"beta\n"}
    expected = {n: hashlib.sha256(b).hexdigest() for n, b in good.items()}
    bad_zip = {"a.csv": b"corrupt\n", "b.csv": good["b.csv"]}
    served_zip, _ = _fake_sources(bad_zip)
    served_raw, sources = _fake_sources(good)
    served = {**served_raw, "zip://x": served_zip["zip://x"]}
    calls: list[str] = []

    def fake_get(url, timeout=0):
        calls.append(url)
        return served[url]

    monkeypatch.setattr(data, "_get", fake_get)
    log: list[str] = []
    paths = data.download(tmp_path, sources, expected, log=log.append)
    assert paths["a.csv"].read_bytes() == b"alpha\n" and paths["b.csv"].read_bytes() == b"beta\n"
    assert calls == ["zip://x", "raw://a.csv"]  # b.csv came from the zip, a.csv needed fallback
    assert any("mismatch" in line for line in log)
    # second call: nothing fetched
    calls.clear()
    data.download(tmp_path, sources, expected, log=log.append)
    assert calls == []


def test_download_raises_when_nothing_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_get", lambda url, timeout=0: b"wrong")
    with pytest.raises(RuntimeError):
        data.download(tmp_path, [{"a.csv": "raw://a"}], {"a.csv": "0" * 64}, log=lambda s: None)


def test_sha256_of_file(tmp_path):
    p = Path(tmp_path / "f")
    p.write_bytes(b"xyz")
    assert data.sha256(p) == hashlib.sha256(b"xyz").hexdigest()
    assert data.verified(p, data.sha256(p)) and not data.verified(p, "0" * 64)
