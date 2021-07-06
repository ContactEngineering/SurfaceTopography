#
# Copyright 2019-2020 Lars Pastewka
#           2019 Michael Röttger
#           2019 Kai Haase
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
import os
from numpy.testing import assert_allclose

from SurfaceTopography.IO.Text import read_xyz

DATADIR = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))),
    'file_format_examples')


def test_read_1d():
    surface = read_xyz(os.path.join(DATADIR, 'example.xyz'))
    assert not surface.is_uniform
    x, y = surface.positions_and_heights()
    assert len(x) > 0
    assert len(x) == len(y)
    assert not surface.is_uniform
    assert surface.dim == 1
    assert not surface.is_periodic


@pytest.mark.parametrize("filename", ['example-2d.xyz', 'example-2d-different-order.xyz'])
def test_read_2d(filename):
    """
    Here the order of points in the input file shouldn't matter.
    """
    surface = read_xyz(os.path.join(DATADIR, filename))
    assert surface.is_uniform
    x, y, z = surface.positions_and_heights()
    assert x.shape == (4, 4)
    assert y.shape == (4, 4)
    assert z.shape == (4, 4)
    assert surface.dim == 2
    assert not surface.is_periodic
    assert_allclose(z,
                    [[1., 1., 1., 1.],
                     [1., 2., 2., 1.],
                     [1., 1., 1., 1.],
                     [1., 1., 1., 1.]])
