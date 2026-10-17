import pytest

from rssi_loc import synthetic


@pytest.fixture(scope="session")
def building():
    return synthetic.make_building(n=400, n_ap=25, seed=0)
