import numpy as np
import pytest

from rssi_loc import evaluate, synthetic
from rssi_loc.features import transform
from rssi_loc.models import KnnLocalizer, MlpLocalizer, Prediction


def _split(fp, n_test):
    tr = slice(0, len(fp) - n_test)
    te = slice(len(fp) - n_test, None)
    sub = lambda s: type(fp)(fp.rssi[s], fp.xy[s], fp.floor[s], fp.building[s], fp.meta.iloc[s])  # noqa: E731
    return sub(tr), sub(te)


def test_knn_recovers_training_points_exactly(building):
    x = transform(building.rssi)
    pred = KnnLocalizer(k=1).fit(x, building.xy, building.floor, building.building).predict(x)
    assert np.allclose(pred.xy, building.xy)
    assert (pred.floor == building.floor).all() and (pred.building == building.building).all()


def test_knn_beats_guessing_the_centre_on_noiseless_building():
    fp = synthetic.make_building(n=1500, n_ap=40, noise_db=0.0, seed=3)
    train, test = _split(fp, 200)
    pred = evaluate.fit_predict(
        KnnLocalizer(k=3), transform(train.rssi), train, transform(test.rssi)
    )
    m = evaluate.summarise(pred, test)
    centre = np.linalg.norm(test.xy - train.xy.mean(axis=0), axis=1).mean()
    assert m.mean_m < 0.4 * centre
    assert m.floor_acc > 0.9 and m.building_acc == 1.0


def test_knn_vote_and_weights_shapes(building):
    x = transform(building.rssi)
    train, test = _split(building, 50)
    for weighted in (True, False):
        m = KnnLocalizer(k=5, weighted=weighted).fit(x[:-50], train.xy, train.floor, train.building)
        p = m.predict(x[-50:])
        assert p.xy.shape == (50, 2) and p.floor.shape == (50,) and p.building.shape == (50,)
        assert set(np.unique(p.floor)) <= {0, 1} and set(np.unique(p.building)) == {0}


def test_mlp_trains_and_predicts(building):
    train, test = _split(building, 80)
    pred = evaluate.fit_predict(
        MlpLocalizer(hidden=(32,), max_iter=60), transform(train.rssi), train, transform(test.rssi)
    )
    m = evaluate.summarise(pred, test)
    assert pred.xy.shape == (80, 2)
    assert m.mean_m < 40  # inside the 80 x 40 m building: better than random
    assert 0 <= m.floor_acc <= 1


def test_summarise_and_cdf():
    truth = synthetic.make_building(n=4, seed=5)
    pred = Prediction(
        truth.xy + np.array([[3, 4], [0, 0], [6, 8], [0, 1]]), truth.floor, truth.building
    )
    m = evaluate.summarise(pred, truth)
    assert m.mean_m == pytest.approx((5 + 0 + 10 + 1) / 4)
    assert m.median_m == pytest.approx(3.0) and m.floor_acc == 1.0 and m.n == 4
    e, p = evaluate.cdf(evaluate.errors_m(pred.xy, truth.xy))
    np.testing.assert_allclose(e, [0, 1, 5, 10])
    assert p[-1] == 1.0 and p[0] == 0.25
    assert m.as_dict()["mean_m"] == 4.0


def test_cross_validate_covers_every_fingerprint(building):
    x = transform(building.rssi)
    metrics, errs = evaluate.cross_validate(lambda: KnnLocalizer(k=3), x, building, n_splits=4)
    assert metrics.n == len(building) == len(errs)
    assert 0 < metrics.median_m < 20


def test_k_sweep_table(building):
    x = transform(building.rssi)
    train, test = _split(building, 60)
    t = evaluate.k_sweep([1, 3], x[:-60], train, x[-60:], test, lambda k: KnnLocalizer(k=k))
    assert list(t["k"]) == [1, 3] and {"mean_m", "floor_acc"} <= set(t.columns)


def test_rssi_repeatability_measures_noise():
    quiet = synthetic.make_building(n=800, n_ap=10, noise_db=0.5, seed=7)
    loud = synthetic.make_building(n=800, n_ap=10, noise_db=4.0, seed=7)
    s_q, s_l = evaluate.rssi_repeatability(quiet), evaluate.rssi_repeatability(loud)
    assert len(s_q) > 0 and len(s_l) > 0
    assert np.median(s_l) > np.median(s_q)
    s_phone = evaluate.rssi_repeatability(loud, same_phone=True)
    assert len(s_phone) > 0 and np.median(s_phone) > np.median(s_q)
