import json

from rssi_loc.cli import main


def test_run_synthetic_end_to_end(tmp_path):
    out, figs = tmp_path / "res", tmp_path / "figs"
    main(
        [
            "run",
            "--synthetic",
            "--out",
            str(out),
            "--figures",
            str(figs),
            "--folds",
            "2",
            "--hidden",
            "16",
        ]
    )
    res = json.loads((out / "results.json").read_text())
    assert res["knn"]["split"]["n"] == 150 and res["mlp"]["cv"]["n"] == 600
    assert len(res["knn"]["k_sweep"]) == 7
    names = sorted(p.name for p in figs.glob("*.png"))
    assert names == [
        "01_error_cdf.png",
        "02_k_sweep.png",
        "03_knn_map.png",
        "03_mlp_map.png",
        "04_rssi_repeatability.png",
    ]
