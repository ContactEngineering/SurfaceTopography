#
# Copyright 2025 Lars Pastewka
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

import io
import os
import struct

import numpy as np
import pytest
from NuMPI import MPI

from SurfaceTopography import read_topography
from SurfaceTopography.IO import TMDReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial funcionalities, please execute with pytest")


def test_read_tmd_filestream(file_format_examples):
    file_path = os.path.join(file_format_examples, 'tmd-1.tmd')

    read_topography(file_path)

    with open(file_path, 'rb') as f:
        read_topography(f)


def test_tmd_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'tmd-1.tmd')

    r = TMDReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 300
    assert ny == 300

    sx, sy = t.physical_sizes
    np.testing.assert_allclose(sx, 18.9566, rtol=1e-3)
    np.testing.assert_allclose(sy, 18.9566, rtol=1e-3)

    # TrueMap stores lengths and heights in millimeters (this is also how
    # Gwyddion's `rmitmd.c` interprets them); a 19 mm field of view with
    # 63 µm pixels and 0.35 mm height range is typical for the cartridge
    # case scans TrueMap is used for.
    assert t.unit == 'mm'
    assert t.info['comment'] == 'Created by TrueMap v6'
    assert t.info['instrument']['vendor'] == 'TrueMap'

    # Verify heights are in reasonable range
    assert t.heights().min() >= 0
    assert t.heights().max() < 1


def test_tmd_format_detection(file_format_examples):
    file_path = os.path.join(file_format_examples, 'tmd-1.tmd')

    from SurfaceTopography.IO import detect_format
    assert detect_format(file_path) == 'tmd'


def _make_tmd(comment, nx, ny, sx, sy, data):
    buffer = b'Binary TrueMap Data File v2.0\r\n\x00'
    buffer += comment.encode('latin-1') + b'\x00'
    buffer += struct.pack('<IIffff', nx, ny, sx, sy, 0.5, 0.25)
    buffer += np.asarray(data, dtype='<f4').tobytes()
    return buffer


def test_tmd_synthetic():
    """Lengths and heights are in mm, data is stored row by row"""
    comment = 'Synthetic test file 1\r\n'  # 23 characters + NUL
    nx, ny = 3, 2
    data = np.arange(nx * ny, dtype=float).reshape(ny, nx)
    buffer = _make_tmd(comment, nx, ny, 1.5, 1.0, data)
    t = TMDReader(io.BytesIO(buffer)).topography()
    assert t.nb_grid_pts == (nx, ny)
    np.testing.assert_allclose(t.physical_sizes, (1.5, 1.0))
    assert t.unit == 'mm'
    np.testing.assert_allclose(t.heights(), data.T)
    assert t.info['comment'] == comment.strip()
