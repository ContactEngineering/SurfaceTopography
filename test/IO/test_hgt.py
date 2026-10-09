#
# Copyright 2020-2021 Lars Pastewka
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

import numpy as np
import pytest

from NuMPI import MPI

from SurfaceTopography.IO.FromFile import HGTReader, read_hgt

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


def test_read(file_format_examples):
    surface = read_hgt(os.path.join(file_format_examples, 'N46E013.hgt'))
    nx, ny = surface.nb_grid_pts
    assert nx == 3601
    assert ny == 3601
    assert surface.is_uniform


def test_orientation_and_voids(tmp_path):
    # SRTM tiles are stored row by row from north to south, each row running
    # from west to east. The first index of the heights (x) must run along a
    # row (west to east), consistent with all other raster readers.
    raw = np.array([[1, 2, 3],
                    [4, -32768, 6],
                    [7, 8, 9]], dtype='>i2')
    fn = tmp_path / 'N00E000.hgt'
    fn.write_bytes(raw.tobytes())

    reader = HGTReader(str(fn))
    t = reader.topography(physical_sizes=(1., 1.))
    assert t.nb_grid_pts == (3, 3)
    h = t.heights()
    np.testing.assert_array_equal(h[:, 0], [1, 2, 3])  # northern row
    np.testing.assert_array_equal(h[0, :], [1, 4, 7])  # western column
    assert t.has_undefined_data
    assert np.ma.getmaskarray(h)[1, 1]
    assert np.ma.getmaskarray(h).sum() == 1


def test_wrong_size(tmp_path):
    fn = tmp_path / 'broken.hgt'
    fn.write_bytes(b'\x00' * 10)
    with pytest.raises(RuntimeError):
        read_hgt(str(fn))
