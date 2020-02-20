import numpy as np
from htof.polynomial import polynomial as poly


def test_taylorvander():
    expected = np.array([[1.,  1.,  0.5],
                         [1.,  2.,  2.],
                         [1.,  3.,  4.5],
                         [1.,  5., 12.5]])

    assert np.allclose(expected, poly.taylorvander(np.array([1, 2, 3, 5]), 2))


def test_taylor_series():
    series = poly.TaylorSeries(np.array([1, 2, 2, 1]))
    for domain in [np.arange(5), np.arange(5, 10), np.arange(-10, 5)]:
        assert np.allclose(series(domain), 1 + 2 * domain + 2 * 1/2 * domain**2 + 1/6 * domain**3)
