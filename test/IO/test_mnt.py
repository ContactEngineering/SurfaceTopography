#
# Copyright 2023-2025 Lars Pastewka
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

from SurfaceTopography import read_topography
from SurfaceTopography.IO import MNTReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


def test_read_filestream(file_format_examples):
    """
    The reader has to work when the file was already opened as binary for
    it to work in topobank.
    """
    file_path = os.path.join(file_format_examples, 'mnt-1.mnt')

    read_topography(file_path)

    with open(file_path, 'rb') as f:
        read_topography(f)

    # This test just needs to arrive here without raising an exception


def test_mnt_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'mnt-1.mnt')

    r = MNTReader(file_path)
    assert len(r.channels) == 1

    t = r.topography()

    # Dimensions calculated from header: 1000 × 576
    # (elements_per_block=18000, rows_per_block=9×2=18, num_blocks=32)
    nx, ny = t.nb_grid_pts
    assert nx == 1000
    assert ny == 576

    # Physical sizes default to pixel count (format doesn't reliably store sizes)
    sx, sy = t.physical_sizes
    np.testing.assert_almost_equal(sx, 1000)
    np.testing.assert_almost_equal(sy, 576)

    # Height data is in raw int16 units (no scale factor extracted from format)
    assert t.unit == 'µm'

    # This file has a small number of masked pixels at extreme int16 values
    # (316 pixels with values near -32766)
    assert t.has_undefined_data
    h = t.heights()
    assert np.sum(h.mask) < 1000  # Very few masked pixels


def test_mnt2_read_filestream(file_format_examples):
    """
    The reader has to work when the file was already opened as binary for
    it to work in topobank.
    """
    file_path = os.path.join(file_format_examples, 'mnt-2.mnt')

    read_topography(file_path)

    with open(file_path, 'rb') as f:
        read_topography(f)


def test_mnt2_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'mnt-2.mnt')

    r = MNTReader(file_path)
    assert len(r.channels) == 1

    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 1280
    assert ny == 960

    # Physical sizes default to pixel count (format doesn't reliably store sizes)
    sx, sy = t.physical_sizes
    np.testing.assert_almost_equal(sx, 1280)
    np.testing.assert_almost_equal(sy, 960)

    assert t.unit == 'µm'

    # File has masked (invalid) regions at corners
    assert t.has_undefined_data

    # Verify valid height data range (corners are masked, valid region is positive)
    h = t.heights()
    assert h.compressed().min() >= 0
    assert h.compressed().max() < 10000
