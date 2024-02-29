#
# Copyright 2020-2021, 2023 Lars Pastewka
#           2021 Michael Röttger
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

import os
import pytest
from numpy.testing import assert_allclose

from NuMPI import MPI

from SurfaceTopography.IO import XYZReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


def test_read_1d(file_format_examples):
    surface = XYZReader(os.path.join(file_format_examples, 'xy-1.txt')).topography()
    assert not surface.is_uniform
    x, y = surface.positions_and_heights()
    assert len(x) > 0
    assert len(x) == len(y)
    assert not surface.is_uniform
    assert surface.dim == 1
    assert not surface.is_periodic


def test_read_1d_2(file_format_examples):
    surface = XYZReader(os.path.join(file_format_examples, 'xy-2.txt')).topography()
    assert not surface.is_uniform
    x, y = surface.positions_and_heights()
    assert len(x) > 0
    assert len(x) == len(y)
    assert not surface.is_uniform
    assert surface.dim == 1
    assert not surface.is_periodic


@pytest.mark.parametrize("filename", ['xyz-1.txt', 'xyz-1-different-order.txt', 'xyz-2.txt'])
def test_read_2d(filename, file_format_examples):
    """
    Here the order of points in the input file shouldn't matter.
    """
    surface = XYZReader(os.path.join(file_format_examples, filename)).topography()
    assert surface.is_uniform
    x, y, z = surface.positions_and_heights()
    assert x.shape == (4, 4)
    assert y.shape == (4, 4)
    assert z.shape == (4, 4)
    assert surface.dim == 2
    assert not surface.is_periodic
    if filename.startswith('xyz-1'):
        assert_allclose(z,
                        [[1., 1., 1., 1.],
                         [1., 2., 2., 1.],
                         [1., 1., 1., 1.],
                         [1., 1., 1., 1.]])


def test_hfm_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'hfm-1.hfm')

    r = XYZReader(file_path)
    t = r.topography()

    nx, = t.nb_grid_pts
    assert nx == 9600

    assert t.is_uniform

    sx, = t.physical_sizes
    assert_allclose(sx, 4.8, rtol=1e-6)

    assert t.unit == 'mm'

    assert_allclose(t.rms_height_from_profile(), 0.000906, rtol=1e-3)

    t = t.detrend('curvature')
    assert_allclose(t.rms_height_from_profile(), 0.000138, rtol=1e-3)


def test_unit_1d(file_format_examples):
    r = XYZReader(os.path.join(file_format_examples, 'xy-1.txt'))
    assert r.channels[0].unit is None
    r = XYZReader(os.path.join(file_format_examples, 'xy-2.txt'))
    assert r.channels[0].unit is None
