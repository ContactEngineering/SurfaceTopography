"""
Tests of the pipeline for topography containers
"""

import numpy as np

from SurfaceTopography import read_container


def test_scale_dependent_statistical_property_uniform(file_format_examples):
    c, = read_container(f'{file_format_examples}/container1.zip')
    s = c.scale_dependent_statistical_property(lambda x, y: np.var(x), n=1, distance=[0.01, 0.1, 1.0, 10], unit='um')
    assert (np.diff(s) < 0).all()
    np.testing.assert_almost_equal(s, [0.0018715281899762592, 0.0006849065620048571, 0.0002991781282532277,
                                       7.224607689277936e-05])

    # Test that specifying distances where no data exists does not raise an exception
    s = c.scale_dependent_statistical_property(lambda x, y: np.var(x), n=1, distance=[0.00001, 1.0, 10000], unit='um')
    assert s[0] is None
    assert s[2] is None


def test_scale_dependent_statistical_property_mixed(file_format_examples):
    c, = read_container(f'{file_format_examples}/container2.zip')
    s = c.scale_dependent_statistical_property(lambda x, y=None: np.var(x), n=1, distance=[0.1, 1.0, 10], unit='um')
    assert (np.diff(s) < 0).all()
