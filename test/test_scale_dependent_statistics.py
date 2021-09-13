#
# Copyright 2021 Lars Pastewka
#
# ### MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import pytest

import numpy as np

from SurfaceTopography import read_container
from SurfaceTopography.Generation import fourier_synthesis


def test_scale_dependent_rms_slope_from_profile():
    t = fourier_synthesis((1024, 1024), (1, 1), 0.8, rms_slope=0.1)

    x, A = t.autocorrelation_from_profile()

    s = t.scale_dependent_statistical_property(lambda x, y: np.var(x), distance=x[1::20])

    np.testing.assert_allclose(2*A[1::20]/x[1::20]**2, s)


def test_scalar_input():
    t = fourier_synthesis((1024, ), (7,), 0.8, rms_slope=0.1)
    s = t.scale_dependent_statistical_property(lambda x: np.var(x), n=1, distance=[0.01, 0.1, 1])
    assert len(s) == 3
    s2 = t.scale_dependent_statistical_property(lambda x: np.var(x), n=1, distance=0.1)
    with pytest.raises(TypeError):
        iter(s2)
    np.testing.assert_almost_equal(s[1], s2)
    s3 = t.scale_dependent_statistical_property(lambda x: np.var(x), n=1, scale_factor=2)
    with pytest.raises(TypeError):
        iter(s3)


def test_nonuniform():
    t = fourier_synthesis((1024, ), (7,), 0.8, rms_slope=0.1, periodic=False)
    p, = t.pixel_size
    s = t.scale_dependent_statistical_property(lambda x: np.var(x), n=1, distance=[p, 4*p, 16*p])
    t2 = t.to_nonuniform()
    t3 = t2.to_uniform(pixel_size=p)
    np.testing.assert_almost_equal(t3.pixel_size, p)
    np.testing.assert_almost_equal(t3.positions()[1] - t3.positions()[0], p)
    s2 = t2.scale_dependent_statistical_property(lambda x: np.var(x), n=1, distance=[p, 4*p, 16*p])
    np.testing.assert_allclose(s, s2, atol=1e-3)


def test_container_uniform(file_format_examples):
    c, = read_container(f'{file_format_examples}/container1.zip')
    s = c.scale_dependent_statistical_property(lambda x, y: np.var(x), n=1, distance=[0.01, 0.1, 1.0, 10], unit='um')
    assert (np.diff(s) < 0).all()
    np.testing.assert_almost_equal(s, [0.0018715281899762592, 0.0006849065620048571, 0.0002991781282532277,
                                       7.224607689277936e-05])

    # Test that specifying distances where no data exists does not raise an exception
    s = c.scale_dependent_statistical_property(lambda x, y: np.var(x), n=1, distance=[0.00001, 1.0, 10000], unit='um')
    assert s[0] is None
    assert s[2] is None


def test_container_mixed(file_format_examples):
    c, = read_container(f'{file_format_examples}/container2.zip')
    s = c.scale_dependent_statistical_property(lambda x, y=None: np.var(x), n=1, distance=[0.1, 1.0, 10], unit='um')
    assert (np.diff(s) < 0).all()


@pytest.mark.skip('Run this if have a one of the big diamond containers download from contact.engineering')
def test_large_container_mixed():
    c, = read_container('/home/pastewka/Downloads/surface.zip')
    distances = np.logspace(np.log10(0.001), np.log10(1000), 11)
    s = c.scale_dependent_statistical_property(lambda x, y=None: np.var(x), n=1, unit='um', distance=distances)
    assert not np.any(np.isnan(s))
