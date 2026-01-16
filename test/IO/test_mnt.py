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

    # Dimensions extracted from TLV tags 0x0007 (width) and 0x0008 (height)
    nx, ny = t.nb_grid_pts
    assert nx == 960
    assert ny == 600

    # Physical sizes extracted from TLV tags 0x0009 and 0x000a (in mm, converted to µm)
    sx, sy = t.physical_sizes
    np.testing.assert_almost_equal(sx, 1777.404, decimal=2)  # 1.7774 mm -> µm
    np.testing.assert_almost_equal(sy, 1110.878, decimal=2)  # 1.1109 mm -> µm

    # Unit is µm
    assert t.unit == 'µm'

    # This file uses pure int32 format without validity channel
    assert not t.has_undefined_data


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

    # Physical sizes extracted from TLV tags 0x0009 and 0x000a (in mm, converted to µm)
    sx, sy = t.physical_sizes
    np.testing.assert_almost_equal(sx, 95.77, decimal=1)  # 0.0958 mm -> µm
    np.testing.assert_almost_equal(sy, 71.81, decimal=1)  # 0.0718 mm -> µm

    assert t.unit == 'µm'

    # File has masked (invalid) regions at corners (zeros mark undefined pixels)
    assert t.has_undefined_data

    # Verify valid height data range
    # Heights are scaled by pixel_scales z value (10 nm/count = 0.01 µm/count)
    # Valid data includes both positive and negative values
    h = t.heights()
    assert h.compressed().min() > -150  # ~-111 µm after scaling
    assert h.compressed().max() < 100  # ~49 µm after scaling

    # Verify masked pixel count (zeros at corners/edges)
    assert 40000 < h.mask.sum() < 50000  # About 45531 masked pixels
